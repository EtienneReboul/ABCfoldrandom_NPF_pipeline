"""
fetch_cdd_sites: the CDD site features of one protein from the InterPro API
(residue-level annotations, ``/api/protein/uniprot/<acc>/?residues``), e.g.
cd17413 MFS_NPF6 "putative chemical substrate binding pocket". Run on the Mac
(network); the JSON travels with data/inputs like the topology.

Every annotated residue is checked against sequence.fasta (same numbering:
UniProt canonical); a mismatch fails the job.

Output: {protein, accession, url, features: [{cdd, name, description,
residues: [resnum ...], aa: "ERTL..."}]}; features = [] when InterPro has none.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/{acc}/?residues"


def fetch(url: str, tries: int = 5) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "npf-pipeline/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 204:
                    return {}
                body = r.read()
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(2 * (i + 1))
    return {}


def read_fasta(path: Path) -> str:
    return "".join(x.strip() for x in path.read_text().splitlines() if not x.startswith(">"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", type=Path, required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    topo = json.loads(a.topology.read_text())
    seq = read_fasta(a.fasta)
    url = API.format(acc=topo["accession"])
    raw = fetch(url)
    feats = []
    for acc, e in raw.items():
        if not isinstance(e, dict) or e.get("source_database") != "cdd":
            continue
        for loc in e.get("locations", []):
            res, aa = [], ""
            for f in loc["fragments"]:
                for k, r in enumerate(range(int(f["start"]), int(f["end"]) + 1)):
                    exp = f["residues"][k] if len(f["residues"]) > k else f["residues"][0]
                    if r > len(seq) or seq[r - 1] != exp:
                        raise ValueError(f"{topo['protein']} {acc}: residue {exp}{r} does not match the sequence "
                                         f"({seq[r - 1] if r <= len(seq) else 'past the end'})")
                    res.append(r)
                    aa += exp
            feats.append({"cdd": acc, "name": e.get("name"), "description": loc.get("description"),
                          "residues": res, "aa": aa})
    a.out.write_text(json.dumps({"protein": topo["protein"], "accession": topo["accession"], "url": url,
                                 "features": feats}, indent=1) + "\n")
    print(f"[cdd_sites] {topo['protein']}: " + (", ".join(f"{f['cdd']} {f['description']} ({len(f['residues'])})"
                                                           for f in feats) or "no CDD site"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
