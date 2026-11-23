{%- set cfg = snakemake.config -%}
{%- set ab = cfg.get("abcfold") or {} -%}
ABCfold-random: Arabidopsis NPF family
======================================

CF-random's sampling recipe (Lee et al. 2025, *Nat Commun* 16:5622: the full
MSA plus randomly subsampled shallow MSAs) run with the open-weight ABCfold
backends ({{ ab.get("backends") | join(", ") }}; AlphaFold3 is left out
because of its weights terms of use) on every NPF: full MSA plus
{{ cfg.get("depths") | length }} depths (query + max_seq + max_extra_seq
sequences drawn at random:
{% for d in cfg.get("depths") %}{{ d[0] }}:{{ d[1] }}{{ ", " if not loop.last }}{% endfor %}),
{{ ab.get("n_seeds") }} seeds x {{ ab.get("number_of_models") }} samples per
backend and depth, {{ ab.get("num_recycles") }} recycles, no templates.
The MSA and TM topology are those of CFrandom_NPF_pipeline, so the two runs
differ only by the predictors.

Sections:

1. **Runs**: GPU time and storage of every job.
2. **Confidence vs MSA depth**, per backend.
3. **TM-helix Ca PCA** over all models of a protein (fit on confident models).
4. **Conformational states**: outward-open / occluded / inward-open read
   from the extremities of the PCA cloud (the V / lambda shape), with
   {{ (cfg.get("tm_states") or {}).get("n_export", 20) }} models per state
   written for ChimeraX to ``results/<protein>/tm_states/structures/``,
   and checked with pyKVFinder: does the CDD substrate pocket of each of
   those models open outside, inside, or not at all? Run on the raw
   models and on OpenMM-minimized copies (side chains move the openings).

Structures are not kept as CIF files: ``scripts/export_models.py`` rebuilds
any model as mmCIF / PDB from the Parquet store (README).
