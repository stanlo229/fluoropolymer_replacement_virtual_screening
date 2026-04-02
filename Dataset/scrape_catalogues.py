#!/usr/bin/env python3
"""
Scrape chemical catalogues (Sigma-Aldrich, Ambeed, Combi-Blocks) via PubChem
for monoalcohols, diols, and monoamines with MW < 500, ≤1 fluorine, no sulfur,
no chiral centres, no phenols, and no anilines.

All three vendors have substance records on PubChem (~250K each). Each SID record
embeds a direct vendor product URL in xref.sburl — no URL construction needed.
Price-per-gram (USD/g) is extracted from the PubChem info table where available.

Output CSV columns: smiles, canonical_smiles, source, link, molecular_weight,
                    compound_type, price_per_gram

Hard filters applied (any match → compound rejected):
  • MW ≥ 500
  • Contains sulfur
  • More than 1 fluorine
  • Any chiral centre (assigned or unassigned)
  • Phenol / aromatic alcohol (OH on aromatic carbon)
  • Aniline / aromatic amine (N on aromatic carbon)

Usage:
    python scrape_catalogues.py --output results.csv
    python scrape_catalogues.py --vendors Ambeed --limit 200   # quick test
    python scrape_catalogues.py --resume                       # continue interrupted run
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Vendor configuration
# ---------------------------------------------------------------------------

VENDORS = {
    "Sigma-Aldrich": "Sigma-Aldrich",
    "Ambeed":        "Ambeed",
    "Combi-Blocks":  "Combi-Blocks",
}

# Short aliases accepted on the command line
VENDOR_ALIASES = {
    "sigma":          "Sigma-Aldrich",
    "aldrich":        "Sigma-Aldrich",
    "sigma-aldrich":  "Sigma-Aldrich",
    "Sigma-Aldrich":  "Sigma-Aldrich",
    "ambeed":         "Ambeed",
    "Ambeed":         "Ambeed",
    "combi":          "Combi-Blocks",
    "combi-blocks":   "Combi-Blocks",
    "Combi-Blocks":   "Combi-Blocks",
}

# ---------------------------------------------------------------------------
# RDKit SMARTS patterns and filter/classify logic
# ---------------------------------------------------------------------------

# C-OH: aliphatic or aromatic carbon bearing an OH (covers phenols),
# but explicitly excludes carboxylic acid OH (O directly bonded to C=O).
_ALCOHOL = Chem.MolFromSmarts("[#6][OX2H;!$(OC=O)]")

# Non-amide / non-sulfonamide / non-cationic / non-imine amine N
# Covers primary R-NH2, secondary R-NH-R', tertiary R3N, aromatic Ar-NH2
_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*);$(N[#6])]"
)

_SULFUR = Chem.MolFromSmarts("[#16]")

# OH directly on any aromatic carbon (phenol, naphthol, hetarene-OH …)
_PHENOL = Chem.MolFromSmarts("[OX2H][c]")

# Amine N directly on any aromatic carbon — same exclusions as _AMINE above
_AROMATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)][c]"
)


def _has_chiral_centre(mol) -> bool:
    return bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


def classify(mol) -> list:
    """
    Return list of matching compound types from {'monoalcohol', 'diol', 'monoamine'},
    or empty list if the compound fails any hard filter (MW, S, F).
    """
    if Descriptors.MolWt(mol) >= 500:
        return []
    if mol.HasSubstructMatch(_SULFUR):
        return []
    if sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms()) > 1:
        return []
    if _has_chiral_centre(mol):
        return []
    if mol.HasSubstructMatch(_PHENOL):
        return []
    if mol.HasSubstructMatch(_AROMATIC_AMINE):
        return []
    oh = len(mol.GetSubstructMatches(_ALCOHOL))
    n  = len(mol.GetSubstructMatches(_AMINE))
    types = []
    if oh == 1:
        types.append("monoalcohol")
    if oh == 2:
        types.append("diol")
    if n == 1:
        types.append("monoamine")
    return types


def build_row(isomeric_smiles: str, source: str, sburl: str, types: list,
              price_per_gram: float | None = None) -> dict | None:
    mol = Chem.MolFromSmiles(isomeric_smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    return {
        "smiles":           isomeric_smiles,
        "canonical_smiles": canon,
        "source":           source,
        "link":             sburl,
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "compound_type":    ",".join(types),
        "price_per_gram":   price_per_gram,
    }

# ---------------------------------------------------------------------------
# HTTP helper with rate-limiting and retry
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})


def http_get(url: str, params: dict = None, retries: int = 5) -> dict:
    """GET request with exponential backoff on 429 / 5xx."""
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=90)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay * 2))
                time.sleep(wait)
                delay *= 2
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed GET {url} after {retries} attempts")

# ---------------------------------------------------------------------------
# Step 1 — Paginated SID listing via NCBI eSearch
# ---------------------------------------------------------------------------

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def esearch_sids(source_name: str, limit: int = None) -> list:
    """Return list of all PubChem substance SIDs for *source_name*."""
    # First call: get total count only
    meta = http_get(_ESEARCH, params={
        "db":      "pcsubstance",
        "term":    f'"{source_name}"[SourceName]',
        "retmax":  1,
        "retmode": "json",
    })
    time.sleep(0.4)

    total = int(meta["esearchresult"]["count"])
    if total == 0:
        print(f"  WARNING: no PubChem substances found for source '{source_name}'",
              file=sys.stderr)
        return []

    if limit is not None:
        total = min(total, limit)

    print(f"  {source_name}: {total:,} substances to retrieve", flush=True)

    sids = []
    retstart = 0
    retmax = 10_000

    with tqdm(total=total, desc=f"  Listing SIDs", unit="SID") as pbar:
        while len(sids) < total:
            fetch_n = min(retmax, total - len(sids))
            data = http_get(_ESEARCH, params={
                "db":       "pcsubstance",
                "term":     f'"{source_name}"[SourceName]',
                "retmax":   fetch_n,
                "retstart": retstart,
                "retmode":  "json",
            })
            time.sleep(0.4)
            batch = [int(x) for x in data["esearchresult"]["idlist"]]
            if not batch:
                break
            sids.extend(batch)
            pbar.update(len(batch))
            retstart += len(batch)

    return sids

# ---------------------------------------------------------------------------
# Step 2 — Batch SID details: extract {sid, cid, sburl, cas, regid}
# ---------------------------------------------------------------------------

_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Patterns for extracting price-per-gram from PC_Substance info tables
_PRICE_LABEL_RE = re.compile(r"\bpric(e|ing)\b|\bcost\b", re.IGNORECASE)
_GRAMS_RE       = re.compile(r"([\d.]+)\s*g\b", re.IGNORECASE)
_DOLLAR_RE      = re.compile(r"\$?\s*([\d,]+\.?\d*)")


def _parse_price_per_gram(info_list: list) -> float | None:
    """
    Scan a PC_Substance info list for price entries.  Returns the cheapest
    USD/g found, or None if no parseable price data is present.

    Handles two common vendor formats:
      • label="Price", name="25 g / USD", value={"sval": "47.50"}
        → price_per_gram = 47.50 / 25 = 1.90
      • label="Price", name="USD/g",       value={"fval": 2.35}
        → price_per_gram = 2.35
    """
    candidates = []

    for item in info_list:
        urn   = item.get("urn", {})
        label = urn.get("label", "")
        if not _PRICE_LABEL_RE.search(label):
            continue

        name = urn.get("name", "")
        val  = item.get("value", {})

        # Extract numeric price
        if "fval" in val:
            price = float(val["fval"])
        elif "sval" in val:
            m = _DOLLAR_RE.search(str(val["sval"]))
            if not m:
                continue
            price = float(m.group(1).replace(",", ""))
        else:
            continue

        # Normalise to per-gram
        m_g = _GRAMS_RE.search(name)
        if m_g:
            grams = float(m_g.group(1))
            if grams > 0:
                candidates.append(price / grams)
        elif re.search(r"/\s*g\b|per\s*g\b", name, re.IGNORECASE):
            candidates.append(price)

    return round(min(candidates), 4) if candidates else None


def _parse_sid_record(substance: dict) -> dict | None:
    """
    Extract {sid, cid, sburl, cas, regid} from one PC_Substances entry.
    Returns None if CID or sburl is absent (mixture, salt, no product link).
    """
    # --- SID ---
    sid = substance.get("sid", {}).get("id")
    if sid is None:
        return None

    # --- CID (from standardised / deposited compound reference) ---
    cid = None
    for comp in substance.get("compound", []):
        cid_val = comp.get("id", {}).get("id", {}).get("cid")
        if cid_val:
            cid = cid_val
            break
    if cid is None:
        return None

    # --- xref fields ---
    xref = substance.get("xref", [])
    sburl = next((x["sburl"] for x in xref if "sburl" in x), None)
    cas   = next((x["rn"]    for x in xref if "rn"    in x), None)
    regid = next((x["regid"] for x in xref if "regid" in x), None)

    if sburl is None:
        return None  # no direct product link → skip

    price_per_gram = _parse_price_per_gram(substance.get("info", []))

    return {"sid": sid, "cid": cid, "sburl": sburl, "cas": cas, "regid": regid,
            "price_per_gram": price_per_gram}


def fetch_sid_details(sids: list, batch_size: int = 100) -> list:
    """
    Return list of {sid, cid, sburl, cas, regid} for all SIDs that have a
    valid CID and a direct product URL.
    """
    results = []
    batches = [sids[i:i + batch_size] for i in range(0, len(sids), batch_size)]

    for batch in tqdm(batches, desc="  Fetching SID details", unit="batch"):
        sid_str = ",".join(str(s) for s in batch)
        url = f"{_PUG}/substance/sid/{sid_str}/JSON"
        try:
            data = http_get(url)
        except Exception as exc:
            print(f"\n  WARNING: SID batch failed ({exc}), skipping", file=sys.stderr)
            time.sleep(2.0)
            continue
        time.sleep(0.2)

        for substance in data.get("PC_Substances", []):
            info = _parse_sid_record(substance)
            if info is not None:
                results.append(info)

    return results

# ---------------------------------------------------------------------------
# Step 3 — Batch compound properties: IsomericSMILES + MolecularWeight
# ---------------------------------------------------------------------------

def fetch_compound_props(cids: list, batch_size: int = 100) -> dict:
    """
    Return dict cid → {smiles, mw}.  Compounds with MW ≥ 500 are dropped here
    to avoid wasting RDKit calls.
    """
    props = {}
    batches = [cids[i:i + batch_size] for i in range(0, len(cids), batch_size)]

    for batch in tqdm(batches, desc="  Fetching properties", unit="batch"):
        cid_str = ",".join(str(c) for c in batch)
        url = (f"{_PUG}/compound/cid/{cid_str}"
               f"/property/IsomericSMILES,CanonicalSMILES,MolecularWeight/JSON")
        try:
            data = http_get(url)
        except Exception as exc:
            print(f"\n  WARNING: property batch failed ({exc}), skipping",
                  file=sys.stderr)
            time.sleep(2.0)
            continue
        time.sleep(0.2)

        for prop in data.get("PropertyTable", {}).get("Properties", []):
            mw = float(prop.get("MolecularWeight", 9999))
            if mw >= 500:
                continue
            cid = prop.get("CID")
            # PubChem returns IsomericSMILES as "SMILES" when there is no
            # stereochemistry, so check both keys.
            smiles = (prop.get("IsomericSMILES")
                      or prop.get("CanonicalSMILES")
                      or prop.get("SMILES", ""))
            if cid and smiles:
                props[cid] = {"smiles": smiles, "mw": mw}

    return props

# ---------------------------------------------------------------------------
# Checkpoint helpers (atomic write via temp file → rename)
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(data))
    else:
        data.to_csv(tmp, index=False)
    tmp.rename(path)


def _load_checkpoint(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)

# ---------------------------------------------------------------------------
# Per-vendor orchestration
# ---------------------------------------------------------------------------

def process_vendor(vendor_name: str, source_name: str, args) -> pd.DataFrame:
    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    safe = vendor_name.replace("/", "_").replace(" ", "_")
    sid_path  = ckpt / f"{safe}_sids.json"
    prop_path = ckpt / f"{safe}_props.csv"

    # ---- Step 1+2: SID listing + SID details --------------------------------
    if args.resume and sid_path.exists():
        print(f"[{vendor_name}] Resuming: loading SID details from {sid_path}",
              flush=True)
        sid_details = _load_checkpoint(sid_path)
    else:
        print(f"\n[{vendor_name}] Step 1: listing SIDs from PubChem …", flush=True)
        sids = esearch_sids(source_name, limit=args.limit)
        if not sids:
            return pd.DataFrame()

        print(f"[{vendor_name}] Step 2: fetching details for {len(sids):,} SIDs …",
              flush=True)
        sid_details = fetch_sid_details(sids)
        n_valid = len(sid_details)
        print(f"[{vendor_name}] {n_valid:,} SIDs have valid CID + product URL",
              flush=True)
        _save_checkpoint(sid_path, sid_details)

    if not sid_details:
        return pd.DataFrame()

    # ---- Step 3: Compound properties ----------------------------------------
    if args.resume and prop_path.exists():
        print(f"[{vendor_name}] Resuming: loading properties from {prop_path}",
              flush=True)
        prop_df = _load_checkpoint(prop_path)
        props = {int(r["cid"]): {"smiles": r["smiles"], "mw": r["mw"]}
                 for _, r in prop_df.iterrows()}
    else:
        unique_cids = list({d["cid"] for d in sid_details})
        print(f"[{vendor_name}] Step 3: fetching properties for "
              f"{len(unique_cids):,} unique CIDs …", flush=True)
        props = fetch_compound_props(unique_cids)
        n_mw = len(props)
        print(f"[{vendor_name}] {n_mw:,} CIDs passed MW < 500 pre-filter",
              flush=True)
        prop_df = pd.DataFrame(
            [{"cid": cid, "smiles": v["smiles"], "mw": v["mw"]}
             for cid, v in props.items()]
        )
        _save_checkpoint(prop_path, prop_df)

    if not props:
        return pd.DataFrame()

    # ---- Step 4+5: RDKit filter + build rows --------------------------------
    print(f"[{vendor_name}] Applying RDKit filters …", flush=True)
    rows = []
    seen_canon: set = set()

    for d in tqdm(sid_details, desc=f"  Filtering [{vendor_name}]", unit="cpd"):
        cid   = d["cid"]
        sburl = d["sburl"]
        if cid not in props or not sburl:
            continue

        smiles = props[cid]["smiles"]
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        types = classify(mol)
        if not types:
            continue

        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            continue
        seen_canon.add(canon)

        row = build_row(smiles, vendor_name, sburl, types, d.get("price_per_gram"))
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[{vendor_name}] → {len(df):,} compounds passed all filters", flush=True)
    return df

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect monoalcohols, diols, monoamines from chemical catalogues via PubChem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default="results.csv",
        help="Output CSV path (default: results.csv)",
    )
    parser.add_argument(
        "--vendors", nargs="+", default=list(VENDORS.keys()),
        metavar="VENDOR",
        help=(
            "Vendors to query (default: all three). "
            "Accepted: sigma, aldrich, Sigma-Aldrich, ambeed, Ambeed, "
            "combi, combi-blocks, Combi-Blocks"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Directory for intermediate checkpoint files (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing checkpoints and skip already-completed vendors.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Process only the first N SIDs per vendor (for quick testing).",
    )
    args = parser.parse_args()

    # Resolve aliases and deduplicate while preserving order
    resolved = []
    for v in args.vendors:
        key = VENDOR_ALIASES.get(v) or VENDOR_ALIASES.get(v.lower())
        if key is None:
            parser.error(
                f"Unknown vendor '{v}'. "
                f"Valid aliases: {sorted(VENDOR_ALIASES.keys())}"
            )
        if key not in resolved:
            resolved.append(key)
    args.vendors = resolved

    all_dfs = []
    for vendor_name in args.vendors:
        source_name = VENDORS[vendor_name]
        df = process_vendor(vendor_name, source_name, args)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No results found.", flush=True)
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined):,} compounds to {args.output}", flush=True)
    print("\nBreakdown by compound_type:")
    # Expand comma-separated types for counting
    from collections import Counter
    type_counts: Counter = Counter()
    for types_str in combined["compound_type"]:
        for t in types_str.split(","):
            type_counts[t.strip()] += 1
    for t, cnt in sorted(type_counts.items()):
        print(f"  {t}: {cnt:,}")
    print("\nBreakdown by source:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
