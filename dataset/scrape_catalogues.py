#!/usr/bin/env python3
"""
Scrape chemical catalogues via PubChem for monoalcohols, diols, monoamines and
dendrons (one focal OH or amine), with no fluorine and no organometallics.
Chiral compounds are allowed since 2026-10-02 (--exclude_chiral restores the old
rule). There is no MW limit by default (--max_mw restores one; the
original 2026-04 library used MW < 500, kept as catalogues_v1_mw500.csv).

Vendors: Sigma-Aldrich, Ambeed, Combi-Blocks, TCI, Thermo Fisher Scientific,
         Oakwood Products, Matrix Scientific.
(Derthon is not present as a PubChem substance source.)

Each SID record embeds a direct vendor product URL in xref.sburl.
Purity (e.g. "≥97%") and price-per-gram (e.g. "$45/5g") are extracted from
the synonyms list on a best-effort basis; many SIDs will have NaN for these.

Output CSV columns: smiles, canonical_smiles, source, link, molecular_weight,
                    compound_type, purity, price_per_gram

Hard filters applied (any match → compound rejected):
  • MW >= --max_mw (only if given; default: no limit)
  • Any fluorine
  • Any metal (organometallics)
  • Any chiral centre (only with --exclude_chiral; allowed by default)

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

import dendrons as dd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Vendor configuration
# ---------------------------------------------------------------------------

VENDORS = {
    "Sigma-Aldrich":           "Sigma-Aldrich",
    "Ambeed":                  "Ambeed",
    "Combi-Blocks":            "Combi-Blocks",
    "TCI":                     "TCI (Tokyo Chemical Industry)",
    "Thermo Fisher Scientific": "Thermo Fisher Scientific",
    "Oakwood Products":        "Oakwood Products",
    "Matrix Scientific":       "Matrix Scientific",
}

# Short aliases accepted on the command line
VENDOR_ALIASES = {
    "sigma":                    "Sigma-Aldrich",
    "aldrich":                  "Sigma-Aldrich",
    "sigma-aldrich":            "Sigma-Aldrich",
    "Sigma-Aldrich":            "Sigma-Aldrich",
    "ambeed":                   "Ambeed",
    "Ambeed":                   "Ambeed",
    "combi":                    "Combi-Blocks",
    "combi-blocks":             "Combi-Blocks",
    "Combi-Blocks":             "Combi-Blocks",
    "tci":                      "TCI",
    "TCI":                      "TCI",
    "tokyo":                    "TCI",
    "thermo":                   "Thermo Fisher Scientific",
    "fisher":                   "Thermo Fisher Scientific",
    "thermofisher":             "Thermo Fisher Scientific",
    "Thermo Fisher Scientific": "Thermo Fisher Scientific",
    "oakwood":                  "Oakwood Products",
    "Oakwood Products":         "Oakwood Products",
    "matrix":                   "Matrix Scientific",
    "Matrix Scientific":        "Matrix Scientific",
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

_METAL = Chem.MolFromSmarts(
    "[Li,Na,K,Rb,Cs,Be,Mg,Ca,Sr,Ba,"
    "Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,"
    "Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,"
    "Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,"
    "Al,Ga,In,Sn,Tl,Pb,Bi,"
    "La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]"
)


MAX_MW: float | None = None      # set from --max_mw in main(); None = no limit
EXCLUDE_CHIRAL = False           # set from --exclude_chiral in main()


def _has_chiral_centre(mol) -> bool:
    return bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


def classify(mol) -> tuple[list, "dd.DendronInfo | None"]:
    """
    Return (types, dendron_info). types is drawn from {'monoalcohol', 'diol',
    'monoamine', 'dendron_alcohol', 'dendron_amine'}; empty if the compound fails
    a hard filter (fragments, MW if --max_mw, F, metals, chirality if --exclude_chiral).

    Ordinary side groups need exactly 1 OH, 2 OH or 1 amine. Dendrons pass on a
    single focal OH / amine (dendrons.py) whatever their periphery.
    """
    if len(Chem.GetMolFrags(mol)) > 1:
        return [], None
    if MAX_MW is not None and Descriptors.MolWt(mol) >= MAX_MW:
        return [], None
    if sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms()) > 0:
        return [], None
    if mol.HasSubstructMatch(_METAL):
        return [], None
    if EXCLUDE_CHIRAL and _has_chiral_centre(mol):
        return [], None
    oh = len(mol.GetSubstructMatches(_ALCOHOL))
    n  = len(mol.GetSubstructMatches(_AMINE))
    types = []
    if oh == 1:
        types.append("monoalcohol")
    if oh == 2:
        types.append("diol")
    if n == 1:
        types.append("monoamine")
    info = dd.analyse(mol)
    if info.is_dendron:
        if info.focal_alcohol is not None:
            types.append("dendron_alcohol")
        if info.focal_amine is not None:
            types.append("dendron_amine")
    usable = "dendron_alcohol" in types or "dendron_amine" in types
    return types, (info if usable else None)


def focal_mapped_smiles(mol, info) -> str:
    """SMILES with the focal OH O as atom map 1 and the focal amine N as map 2."""
    m = Chem.Mol(mol)
    if info.focal_alcohol is not None:
        m.GetAtomWithIdx(info.focal_alcohol).SetAtomMapNum(1)
    if info.focal_amine is not None:
        m.GetAtomWithIdx(info.focal_amine).SetAtomMapNum(2)
    return Chem.MolToSmiles(m)


def build_row(isomeric_smiles: str, source: str, sburl: str, types: list,
              purity: float | None = None,
              price_per_gram: float | None = None,
              dendron: "dd.DendronInfo | None" = None) -> dict | None:
    mol = Chem.MolFromSmiles(isomeric_smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    return {
        "is_dendron":         dendron is not None,
        "dendron_family":     dendron.family if dendron else "",
        "dendron_generation": dendron.generation if dendron else 0,
        "focal_mapped_smiles": focal_mapped_smiles(mol, dendron) if dendron else "",
        "smiles":           isomeric_smiles,
        "canonical_smiles": canon,
        "source":           source,
        "link":             sburl,
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "compound_type":    ",".join(types),
        "purity":           purity,
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

# ---- Purity extraction from synonyms ----
# Matches: ≥97%, >=98.5%, >99%, 97+%, 97% min, 95% pure, purity: 97%
_PURITY_RE = re.compile(
    r'(?:>=|≥|>|≧)\s*(\d{2,3}(?:\.\d+)?)\s*%'
    r'|(\d{2,3}(?:\.\d+)?)\+\s*%'
    r'|(\d{2,3}(?:\.\d+)?)\s*%\s*(?:min\.?|pure|purity)'
    r'|purity[:\s]+(\d{2,3}(?:\.\d+)?)\s*%',
    re.IGNORECASE,
)

# ---- Price-per-gram extraction from synonyms ----
# Matches: $45.00 / 5 g, USD 12.50/1g, 47.50 USD / 25 g, 9.90/g
_PRICE_RE = re.compile(
    r'(?:USD|\$)\s*([\d,]+\.?\d*)\s*/\s*([\d.]+)\s*(g|kg|mg)'
    r'|([\d,]+\.?\d*)\s*(?:USD|\$)\s*/\s*([\d.]+)\s*(g|kg|mg)'
    r'|(?:USD|\$)\s*([\d,]+\.?\d*)\s*/\s*g\b',
    re.IGNORECASE,
)

_MASS_SCALE = {"mg": 0.001, "g": 1.0, "kg": 1000.0}


def _parse_purity_and_price(synonyms: list) -> tuple:
    """
    Scan the synonyms list for purity (%) and price-per-gram (USD/g).
    Returns (purity_float_or_None, price_per_gram_float_or_None).
    Takes the first purity found and the cheapest price found.
    """
    purity = None
    price_candidates = []

    for syn in synonyms:
        s = str(syn)

        # Purity
        if purity is None:
            m = _PURITY_RE.search(s)
            if m:
                val = next(g for g in m.groups() if g is not None)
                purity = float(val)

        # Price per gram
        m = _PRICE_RE.search(s)
        if m:
            g = m.groups()
            # pattern 1: USD price / qty unit
            if g[0] and g[1] and g[2]:
                price = float(g[0].replace(",", ""))
                qty   = float(g[1]) * _MASS_SCALE.get(g[2].lower(), 1.0)
                if qty > 0:
                    price_candidates.append(price / qty)
            # pattern 2: qty unit / USD price
            elif g[3] and g[4] and g[5]:
                price = float(g[3].replace(",", ""))
                qty   = float(g[4]) * _MASS_SCALE.get(g[5].lower(), 1.0)
                if qty > 0:
                    price_candidates.append(price / qty)
            # pattern 3: USD price / g (no explicit quantity)
            elif g[6]:
                price_candidates.append(float(g[6].replace(",", "")))

    price_per_gram = round(min(price_candidates), 4) if price_candidates else None
    return purity, price_per_gram


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

    synonyms = substance.get("synonyms", [])
    purity, price_per_gram = _parse_purity_and_price(synonyms)

    return {"sid": sid, "cid": cid, "sburl": sburl, "cas": cas, "regid": regid,
            "purity": purity, "price_per_gram": price_per_gram}


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
    Return dict cid → {smiles, mw}.  With --max_mw, heavier compounds are
    dropped here to avoid wasting RDKit calls.
    """
    props = {}
    batches = [cids[i:i + batch_size] for i in range(0, len(cids), batch_size)]
    if not batches:
        return props

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
            if MAX_MW is not None and mw >= MAX_MW:
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
    prop_path = ckpt / f"{safe}_props.csv"            # 2026-04 run: MW < 500 only
    prop_all_path = ckpt / f"{safe}_props_all.csv"    # no MW limit

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
    if MAX_MW is None and args.resume and prop_all_path.exists():
        print(f"[{vendor_name}] Resuming: loading properties from {prop_all_path}", flush=True)
        prop_df = _load_checkpoint(prop_all_path)
        props = {int(r["cid"]): {"smiles": r["smiles"], "mw": r["mw"]}
                 for _, r in prop_df.iterrows()}
    elif MAX_MW is None and args.resume and prop_path.exists():
        # top-up: the old checkpoint dropped MW >= 500; fetch only what is missing
        prop_df = _load_checkpoint(prop_path)
        props = {int(r["cid"]): {"smiles": r["smiles"], "mw": r["mw"]}
                 for _, r in prop_df.iterrows()}
        missing = sorted({d["cid"] for d in sid_details} - set(props))
        print(f"[{vendor_name}] Step 3 top-up: fetching {len(missing):,} CIDs "
              f"missing from the MW<500 checkpoint …", flush=True)
        extra = fetch_compound_props(missing)
        print(f"[{vendor_name}] {len(extra):,} additional CIDs fetched", flush=True)
        props.update(extra)
        prop_df = pd.DataFrame([{"cid": cid, "smiles": v["smiles"], "mw": v["mw"]}
                                for cid, v in props.items()])
        _save_checkpoint(prop_all_path, prop_df)
    elif args.resume and prop_path.exists():
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
        print(f"[{vendor_name}] {n_mw:,} CIDs with properties"
              + (f" (MW < {MAX_MW})" if MAX_MW is not None else ""), flush=True)
        prop_df = pd.DataFrame(
            [{"cid": cid, "smiles": v["smiles"], "mw": v["mw"]}
             for cid, v in props.items()]
        )
        _save_checkpoint(prop_all_path if MAX_MW is None else prop_path, prop_df)

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

        types, dendron = classify(mol)
        if not types:
            continue

        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            continue
        seen_canon.add(canon)

        row = build_row(smiles, vendor_name, sburl, types,
                        d.get("purity"), d.get("price_per_gram"), dendron)
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
            "Vendors to query (default: all). "
            "Accepted aliases: sigma/aldrich, ambeed, combi, tci/tokyo, "
            "thermo/fisher, oakwood, matrix. "
            "Note: Derthon is not a PubChem substance source."
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
        "--max_mw", type=float, default=None,
        help="Drop compounds with MW >= this (default: no limit; the 2026-04 run used 500).",
    )
    parser.add_argument(
        "--exclude_chiral", action="store_true",
        help="Drop compounds with chiral centres (the pre-2026-10 rule; default: allowed).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Process only the first N SIDs per vendor (for quick testing).",
    )
    args = parser.parse_args()
    global MAX_MW, EXCLUDE_CHIRAL
    MAX_MW = args.max_mw
    EXCLUDE_CHIRAL = args.exclude_chiral

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

    _VENDOR_PRIORITY = {
        "Sigma-Aldrich": 0,
        "Ambeed": 1,
        "Thermo Fisher Scientific": 2,
        "TCI": 3,
        "Combi-Blocks": 4,
        "Oakwood Products": 5,
        "Matrix Scientific": 6,
    }
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["_priority"] = combined["source"].map(_VENDOR_PRIORITY).fillna(99).astype(int)
    combined.sort_values("_priority", inplace=True)
    combined.drop_duplicates(subset="canonical_smiles", keep="first", inplace=True)
    combined.drop(columns=["_priority"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
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
