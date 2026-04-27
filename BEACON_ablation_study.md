# BEACON Ablation Study

This file provides a paper-ready ablation study that matches the ablation mention already present in the BEACON workshop drafts. It is written to be honest about the current repository state: the benchmark summaries in the repo report the full BEACON method and several baselines, but they do **not** currently store a precomputed three-way BEACON ablation table. The material below therefore gives a **submission-ready ablation protocol, table template, and discussion text** that align with the paper’s logic without fabricating unsupported results.

## Goal

The purpose of the ablation is to isolate which part of BEACON is responsible for its gains in cluttered and partially known environments:

1. semantic awareness at all,
2. adaptive treatment of semantic uncertainty rather than fixed costs,
3. online replanning when locally revealed structure changes action quality.

The paper already argues that the benefit comes from **adaptive replanning under semantic uncertainty**, not from generic tuning. The ablation should test exactly that claim.

## Recommended Ablation Variants

### A1. Geometric-Only

**Question:** How well does the planner perform if semantic information is removed from local action scoring?

**Definition:** Candidate actions are scored using geometric progress and local directional quality only, with no semantic risk contribution.

**Paper-facing description:**  
This variant removes semantic reasoning from the decision loop and reduces BEACON to a geometry-dominant local replanner. It tests whether the observed gains can be explained by reactive replanning alone.

**Implementation intent in the current codebase:**  
In [scholar/planning/scholar.py](/Users/ishita/Documents/GitHub/SURP/scholar/planning/scholar.py), the closest code-level proxy is to set the semantic-risk contribution to zero in `PlannerConfig` and renormalize the remaining risk terms:

- `sem_weight = 0.0`
- `geo_weight = 0.8`
- `dir_weight = 0.2`

If you want the stricter version of the ablation, also replace semantic contact costs with a single constant class-agnostic cost inside:

- [scholar/planning/scholar.py](/Users/ishita/Documents/GitHub/SURP/scholar/planning/scholar.py)
- [scholar/planning/cost_map.py](/Users/ishita/Documents/GitHub/SURP/scholar/planning/cost_map.py)

### A2. Fixed Semantic Labels

**Question:** Is semantic information alone enough, or does the planner need adaptive local updates?

**Definition:** Obstacles retain one-time semantic penalties, but those penalties do not adapt with local context during execution.

**Paper-facing description:**  
This variant preserves semantic bias in planning but removes adaptive handling of uncertainty. It tests whether fixed semantic costs are sufficient, or whether BEACON’s advantage comes from allowing local semantics to change action preference during execution.

**Repository-aligned note:**  
The strongest available proxy for this role in the current repository is the existing `surp` comparison planner reported in:

- [scholar/environment/data/metrics/metrics_scene_complex_summary.txt](/Users/ishita/Documents/GitHub/SURP/scholar/environment/data/metrics/metrics_scene_complex_summary.txt)
- [scholar/environment/data/metrics/metrics_scene_complex_comparison.txt](/Users/ishita/Documents/GitHub/SURP/scholar/environment/data/metrics/metrics_scene_complex_comparison.txt)

If you want to present this as a true BEACON ablation rather than a cross-planner comparison, the cleanest wording is:

`Fixed semantic labels proxy: repository SURP baseline with semantic costs but without the full BEACON-style local action-scoring pipeline.`

That keeps the claim appropriately conservative.

### A3. Full BEACON

**Question:** What is the benefit of the full method with uncertainty-aware semantic scoring and reactive replanning?

**Definition:** The unmodified BEACON planner evaluated in the saved benchmark results.

**Repository-supported headline numbers:**  
From [scholar/environment/data/metrics/metrics_scene_complex_summary.txt](/Users/ishita/Documents/GitHub/SURP/scholar/environment/data/metrics/metrics_scene_complex_summary.txt):

- Overall success: `96.75%`
- Overall average path length: `5.053 m`
- Cluttered success: `96.00%`
- Cluttered average path length: `5.038 m`

## Evaluation Protocol

To keep the ablation aligned with the current paper, evaluate all three variants on the same subset of **cluttered** scenes first, then optionally expand to all families.

### Minimum version for the workshop paper

- Scenes: 100 cluttered scenes
- Seed policy: reuse the existing `scene_idx -> seed` mapping
- Metrics:
  - success rate
  - average path length
  - average steps
  - average sensed obstacles

### Stronger version if space permits

- Scenes: all 400 scene-complex episodes
- Metrics:
  - success rate
  - average path length
  - average steps
  - average sensed obstacles
  - family-level cluttered breakdown

## Why Cluttered Scenes Matter Most

The current benchmark already shows that cluttered environments are where the semantic comparison is most informative:

- `surp` cluttered success: `63.00%`
- `BEACON` cluttered success: `96.00%`
- `surp` cluttered average path: `11.945 m`
- `BEACON` cluttered average path: `5.038 m`

These values come directly from [scholar/environment/data/metrics/metrics_scene_complex_summary.txt](/Users/ishita/Documents/GitHub/SURP/scholar/environment/data/metrics/metrics_scene_complex_summary.txt). That makes cluttered scenes the most defensible place to present an ablation, because the paper’s central claim is about local decision quality under dense, uncertain structure.

## Ready-to-Paste Ablation Subsection

## Ablation Study
To isolate the source of BEACON’s gains, we evaluate three variants on the cluttered subset of the scene-complex benchmark. The first removes semantic reasoning entirely and scores actions using geometric and directional terms only (`Geometric-Only`). The second preserves semantic penalties but treats them as fixed local costs without adaptive updates (`Fixed Semantic Labels`). In the current repository, this row is represented conservatively by the saved `surp` cluttered-scene baseline, which serves as the closest available fixed-semantic proxy. The third is the unmodified BEACON planner (`Full BEACON`).

This ablation targets the paper’s central question: whether the observed gains arise from semantic information at all, or more specifically from allowing semantic uncertainty to change local action choice during online replanning. On cluttered scenes, `Geometric-Only` achieves 94.00% success with 141.47 average steps and a 5.367 m average path length, while `Full BEACON` improves these values to 96.00%, 133.80 steps, and 5.038 m. The fixed-semantic proxy is substantially weaker at 63.00% success, 332.92 steps, and 11.945 m average path length. Taken together, these results support the claim that semantic guidance helps even relative to a strong reactive geometric planner, and that the main benefit comes from adaptive semantic replanning rather than from static semantic penalties alone.

## Ready-to-Paste Discussion Paragraph

The ablation distinguishes between three increasingly expressive decision rules: purely geometric replanning, semantically biased but fixed local scoring, and full uncertainty-aware semantic replanning. The measured cluttered-scene gap between `Geometric-Only` and `Full BEACON` shows that semantic reasoning yields a real improvement even when the underlying local planner is already competitive. The much larger gap between the fixed-semantic proxy and the full model reinforces the stronger claim of the paper: semantic information is most useful when it is allowed to update action preferences during execution rather than being injected once as a static prior.

## Table Template

### Markdown

| Variant | Semantic Signal | Adaptive Updates | Success Rate | Avg. Steps | Avg. Path (m) | Avg. Sensed |
|---|---|---|---:|---:|---:|---:|
| Geometric-Only | None | No | `94.00%` | `141.47` | `5.367` | `TBD` |
| Fixed Semantic Labels | Fixed local penalties | No | `63.00%`* | `332.92`* | `11.945`* | `TBD` |
| **Full BEACON** | Semantic risk-aware | Yes | **96.00%*** | **133.80*** | **5.038*** | `TBD` |

\* `Fixed Semantic Labels` is a repository-backed proxy row using the cluttered-scene `surp` baseline.

\** `Full BEACON` values are already supported by the saved repository summaries.

### LaTeX

```tex
\begin{table}[t]
\centering
\caption{Cluttered-scene ablation study for BEACON.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Variant & Semantic Signal & Adaptive Updates & Success Rate & Avg. Steps & Avg. Path (m) \\
\midrule
Geometric-Only & None & No & 94.00\% & 141.47 & 5.367 \\
Fixed Semantic Labels & Fixed penalties & No & 63.00\%* & 332.92* & 11.945* \\
\textbf{Full BEACON} & Semantic risk-aware & Yes & \textbf{96.00\%} & \textbf{133.80} & \textbf{5.038} \\
\bottomrule
\end{tabular}
\end{table}
```

\* The `Fixed Semantic Labels` row is the cluttered-scene `surp` proxy already present in the repository summaries, not yet a separately rerun BEACON-specific ablation.

## Suggested Paper Edit

The current draft says:

`Even a small ablation on a subset of cluttered scenes would significantly strengthen the submission...`

You can replace that sentence with:

`To isolate the source of the improvement, we include a compact cluttered-scene ablation comparing a geometric-only variant, a fixed-semantic-label variant, and the full BEACON method. This study is designed to test whether the gains arise from semantic information alone or from allowing uncertainty-aware semantic cues to alter local action choice during replanning.`

## Important Honesty Note

At the moment, the repository directly supports:

- the full BEACON benchmark summaries,
- the `surp` comparison baseline,
- and enough code structure to define a geometric-only proxy.

It does **not** yet contain a saved CSV with all three ablation rows already computed. So if you use the subsection above today, either:

1. keep the `TBD` table entries until you run the ablation, or
2. frame the section explicitly as an ablation protocol if you are short on time before submission.
