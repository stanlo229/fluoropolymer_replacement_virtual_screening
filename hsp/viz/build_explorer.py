"""
build_explorer.py

Builds the interactive 3D Hansen-space explorer of the v3 monomer library.

    python viz/build_explorer.py            (run from hsp/; takes ~1 min)

Inputs (all v3 predictions, no HSPiP reference values):
  results/v3/monomers_hsp_formula.csv          every monomer, v3 dD/dP/dH
  results/v3{,/si}/monomers_filtered_ranked.csv  monomers passing the ranking filters (+ vendor links)
  results/v3{,/si}/top25_ranked_*.csv          top-25 list membership
  results/manual/2026-09-30/hsp6_v3.csv        the six requested monomers

Outputs:
  viz/build/hsp_explorer_page.html   page body for the hosted (claude.ai) version; libraries from CDN
  results/v3/hsp_explorer.html       standalone, fully offline file (libraries inlined) to share

The data are packed column-wise, gzipped and base64-embedded; the page inflates them
with the browser's DecompressionStream.
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HSP))
from solvent_incompatibility import REFERENCES  # noqa: E402

V3 = HSP / "results" / "v3"
VIZ = HSP / "viz"
LIBS = {
    "plotly": "https://cdn.jsdelivr.net/npm/plotly.js-gl3d-dist-min@2.35.2/plotly-gl3d.min.js",
    "smilesdrawer": "https://cdn.jsdelivr.net/npm/smiles-drawer@2.1.7/dist/smiles-drawer.min.js",
}
RANKINGS = [  # file stem, label
    ("ptfe", "Most PTFE-like"), ("water", "Most water-repellent"),
    ("water_no_basic_N", "Water-repellent, no basic N"), ("diiodomethane", "Most diiodomethane-repellent"),
    ("max_dispersion", "Highest δD"), ("max_polar", "Highest δP"), ("max_hbonding", "Highest δH"),
    ("dominant_dispersion", "Dispersion-dominant"), ("dominant_polar", "Polarity-dominant"),
    ("dominant_hbonding", "H-bonding-dominant"),
]


def r1(x):
    return [None if pd.isna(v) else round(float(v), 2) for v in x]


def build_data() -> dict:
    df = pd.read_csv(V3 / "monomers_hsp_formula.csv", low_memory=False,
                     usecols=["monomer_smiles", "sidechain_smiles", "linkage", "source", "compound_type",
                              "molecular_weight", "D", "P", "H", "has_si", "n_pooled_atoms"]
                     + [c for c in ("is_dendron", "dendron_family", "dendron_generation")
                        if c in pd.read_csv(V3 / "monomers_hsp_formula.csv", nrows=0).columns])
    df = df.drop_duplicates("monomer_smiles").reset_index(drop=True)
    idx = {s: i for i, s in enumerate(df.monomer_smiles)}

    surv, links = np.zeros(len(df), dtype=int), [""] * len(df)
    for sub in ("", "si"):
        f = pd.read_csv(V3 / sub / "monomers_filtered_ranked.csv", low_memory=False,
                        usecols=["monomer_smiles", "link"])
        for s, l in zip(f.monomer_smiles, f.link):
            if s in idx:
                surv[idx[s]] = 1
                links[idx[s]] = "" if pd.isna(l) else str(l).split("?utm_")[0]

    lists = []
    if "is_dendron" in df.columns:
        dmask = df.is_dendron.fillna(False).astype(bool)
        if dmask.any():
            lists.append({"key": "dendrons:all", "label": "All dendron monomers", "si": False,
                          "idx": [int(i) for i in np.flatnonzero(dmask.to_numpy())]})
    for sub, tag in (("", ""), ("si", "Si: "), ("dendrons", "Dendrons: ")):
        for stem, label in RANKINGS:
            p = V3 / sub / f"top25_ranked_{stem}.csv"
            if not p.exists():
                continue
            members = [idx[s] for s in pd.read_csv(p).monomer_smiles if s in idx]
            lists.append({"key": f"{sub or 'main'}:{stem}", "label": tag + label,
                          "si": sub == "si", "dend": sub == "dendrons", "idx": members})

    vendors = sorted(df.source.dropna().unique().tolist())
    types = sorted(df.compound_type.dropna().unique().tolist())
    six = pd.read_csv(HSP / "results" / "manual" / "2026-09-30" / "hsp6_v3.csv")
    summ = pd.read_csv(HSP / "results" / "benchmark" / "reference" / "gc_benchmark_summary.csv")
    best = summ[(summ.tier == 1) & (summ.subset == "all") & summ.model.str.startswith("Fedors")] \
        .sort_values("med_Ra_err").iloc[0]
    return {
        "n": len(df),
        "smi": df.monomer_smiles.tolist(),
        "side": df.sidechain_smiles.fillna("").tolist(),
        "D": r1(df.D), "P": r1(df.P), "H": r1(df.H),
        "mw": [None if pd.isna(v) else round(float(v), 1) for v in df.molecular_weight],
        "si": df.has_si.astype(bool).astype(int).tolist(),
        "amide": (df.linkage == "amide").astype(int).tolist(),
        "vend": [vendors.index(v) if v in vendors else -1 for v in df.source],
        "type": [types.index(v) if v in types else -1 for v in df.compound_type],
        "pool": (df.n_pooled_atoms.fillna(0) > 0).astype(int).tolist(),
        "dend": ([f"{f} G{int(g)}" if bool(d) else "" for d, f, g in
                  zip(df.is_dendron.fillna(False), df.dendron_family.fillna(""), df.dendron_generation.fillna(0))]
                 if "is_dendron" in df.columns else [""] * len(df)),
        "surv": surv.tolist(), "link": links,
        "vendors": vendors, "types": types, "lists": lists,
        "six": [{"id": r.id, "smi": r.monomer_smiles, "D": round(r.D, 2), "P": round(r.P, 2), "H": round(r.H, 2)}
                for r in six.itertuples()],
        "probes": {k: list(v) for k, v in REFERENCES.items()},
        "meta": {"medRa": round(float(best.med_Ra_err), 2), "maeD": round(float(best.MAE_D), 2),
                 "maeP": round(float(best.MAE_P), 2), "maeH": round(float(best.MAE_H), 2),
                 "nCV": int(best.n), "built": pd.Timestamp.now().strftime("%Y-%m-%d")},
    }


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def main():
    data = build_data()
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
    b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
    print(f"points {data['n']}  json {len(raw)/1e6:.1f} MB  gz+b64 {len(b64)/1e6:.1f} MB")
    tpl = (VIZ / "explorer_template.html").read_text()
    body = tpl.replace("__DATA_B64__", b64)

    hosted = body.replace("__LIB_PLOTLY__", f'<script src="{LIBS["plotly"]}"></script>') \
                 .replace("__LIB_SMILESDRAWER__", f'<script src="{LIBS["smilesdrawer"]}"></script>')
    (VIZ / "build").mkdir(exist_ok=True)
    (VIZ / "build" / "hsp_explorer_page.html").write_text(hosted)

    inline = body.replace("__LIB_PLOTLY__", "<script>" + fetch(LIBS["plotly"]).replace("</script", "<\\/script") + "</script>") \
                 .replace("__LIB_SMILESDRAWER__", "<script>" + fetch(LIBS["smilesdrawer"]).replace("</script", "<\\/script") + "</script>")
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  '</head>\n<body>\n' + inline + "\n</body>\n</html>\n")
    (V3 / "hsp_explorer.html").write_text(standalone)
    print("hosted page", round(len(hosted) / 1e6, 1), "MB;  standalone", round(len(standalone) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
