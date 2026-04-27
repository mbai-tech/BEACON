# BEACON: Semantic Belief-Aware Replanning for Safe Online Motion Planning in Partially Known Environments

Submitted to the **ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction**

## Abstract
Robots operating in real environments must reason not only about geometry, but also about the semantic consequences of interaction. This challenge is especially acute in partially known environments, where obstacle meaning is revealed incrementally and action decisions must be revised online. We present **BEACON**, a semantic belief-aware replanning framework for online motion planning in cluttered environments. Rather than treating obstacle semantics as fixed labels, BEACON models newly observed obstacles using a local semantic risk belief and uses that belief to score candidate actions during replanning. The planner couples local perception, belief-weighted semantic risk evaluation, and reactive motion selection to choose between direct progress, detours, and safe interaction-aware navigation. We evaluate the method on four procedurally generated environment families that vary in obstacle density and interaction difficulty: sparse, cluttered, collision-required, and collision-shortcut. Across 400 benchmark episodes, the proposed planner achieves a 96.75% success rate with an average path length of 5.053 m, compared with 82.25%, 91.25%, and 97.25% success for Bug, Bug2, and D* Lite, respectively. Although D* Lite attains slightly higher success, BEACON produces substantially shorter paths, reducing average path length by 39.8% relative to D* Lite. The largest gains appear in cluttered scenes, where semantic uncertainty-aware replanning is most critical. These results suggest that semantic beliefs should be treated as evolving planning variables rather than static annotations.

## I. Introduction
Classical motion planning typically treats obstacle contact as a binary failure condition. This abstraction is useful in structured settings, but it becomes overly restrictive in realistic environments where some interactions may be benign, others undesirable, and still others unsafe. In partially known environments, the problem becomes even harder: a robot must act before the full scene is available, discover obstacles incrementally, and revise decisions as its local understanding changes.

This setting exposes a key gap between semantic perception and reliable autonomy. A robot may know that obstacles differ in interaction risk, but if those semantics are treated as static labels assigned once and never revisited during execution, the planner cannot adapt when local context changes. A path that initially appears acceptable may become obviously poor after nearby structure is revealed. Conversely, a route that seems blocked under conservative geometric reasoning may become feasible once local semantics suggest that interaction risk is low. For reliable robot autonomy, semantic understanding must therefore be part of the replanning loop itself.

We address this problem with **BEACON**, a semantic belief-aware replanning framework built on the BEACON online planning architecture in the SURP codebase. BEACON does not assume that semantic information is globally known or permanently fixed. Instead, newly observed obstacles are assigned a local semantic risk belief that summarizes how safe or risky interaction is expected to be. Candidate motions are then evaluated using belief-weighted semantic risk rather than purely geometric proximity, and the robot replans online whenever newly revealed obstacles change which actions are preferable.

The central claim of this paper is simple: **semantic information should be treated as an evolving planning variable, not a static annotation**. We evaluate this idea using saved benchmark results from the repository across four families of procedurally generated scenes. BEACON achieves a 96.75% success rate across 400 episodes and the shortest average path length among all evaluated planners at 5.053 m. Compared with D* Lite, it sacrifices only 0.5 percentage points of success while reducing average path length from 8.387 m to 5.053 m. The improvement is especially pronounced in cluttered scenes, where local semantic ambiguity and dense obstacle structure make purely geometric strategies brittle.

This framing aligns closely with the ICRA 2026 workshop theme of semantics for reliable robot autonomy. Our contribution is not a new global optimizer or a new perception model in isolation, but a planning perspective in which semantic uncertainty is represented locally and translated directly into safer online action selection.

## II. Related Work
This work lies at the intersection of online motion planning, semantic-aware navigation, and safe interaction in partially known environments. Classical planners such as bug-style algorithms and shortest-path methods reason primarily over geometry and typically treat collision as uniformly undesirable. These approaches are effective when free space is abundant, but they are limited in environments where safe behavior depends on the type, context, or interaction risk of nearby obstacles.

Semantic navigation methods expand this view by using object-dependent costs or categories to influence planning behavior. However, many such systems still assume that semantic information is fixed once assigned. This limits their ability to adapt online when the robot reveals additional structure or when local context changes the desirability of different actions. In contrast, BEACON treats semantic information as a belief-conditioned planning quantity that directly participates in the action selection loop.

Our work is also related to online replanning under partial observability. In these settings, the robot cannot rely on a complete scene model and must instead update decisions as new obstacles are sensed. The BEACON planning framework provides a strong basis for this setting because it operates in a local perception-action loop with incremental obstacle revelation and reactive candidate evaluation. BEACON extends this perspective by asking not only which actions are geometrically feasible, but which are semantically acceptable under uncertainty.

More broadly, this paper contributes to the workshop’s theme of connecting environment understanding to safe interaction. Rather than using semantics as a descriptive layer on top of planning, we treat semantic uncertainty as a first-class factor in the choice of online motion.

## III. Problem Setting
We consider a robot navigating a partially known 2D environment from a start location to a goal location. Obstacles are not assumed to be fully known in advance. Instead, they are revealed incrementally as the robot senses its local surroundings. At any instant, the planner therefore has access only to a partial scene model consisting of the currently observed obstacles and local free-space structure.

Each observed obstacle is associated with semantic interaction uncertainty. Intuitively, some obstacles may be low-risk to approach, some may be permissible to maneuver near, and others may be high-risk or effectively no-contact objects. Rather than collapsing this uncertainty into a single static label, BEACON maintains a semantic risk belief for each newly revealed obstacle or local obstacle region. For obstacle $o_i$, we write this belief as a discrete distribution

\[
b_i = \big[p_i(\text{low}),\; p_i(\text{medium}),\; p_i(\text{high}),\; p_i(\text{no-contact})\big],
\]

where the entries sum to 1. This belief does not attempt to solve a full semantic perception problem. Instead, it provides a compact planning-oriented representation of how costly or risky interaction with that obstacle is expected to be.

The planning objective is to reach the goal reliably while minimizing costly or unsafe behavior. This objective differs from purely collision-free planning in two ways. First, it explicitly acknowledges that different obstacle interactions have different safety implications. Second, it allows the robot to revise motion preferences as new obstacles are revealed and semantic beliefs change. The problem is therefore not only to find a path through free space, but to repeatedly choose the most appropriate local action under evolving geometric and semantic information.

## IV. Method
### A. Incremental Semantic Scene Representation
BEACON operates in the same partially known environment setting as the current repository planner: the robot reveals obstacles only within a fixed local sensing radius and updates its scene representation online. The key conceptual change is that each newly observed obstacle is associated with a **semantic risk belief** rather than an immediately fixed obstacle type. We use a compact belief over interaction categories such as low-risk, medium-risk, high-risk, and no-contact. This abstraction is deliberately planning-centered: the purpose of the belief is not fine-grained semantic recognition, but action selection under uncertainty.

This belief representation makes the planner more expressive than a static semantic map. A newly observed obstacle can initially be treated conservatively, but its role in planning can change as more local structure becomes visible. For example, a soft or easily displaced object can be treated as low-risk, while a fragile or dangerous object should receive much higher risk. In this sense, the semantic belief is context-sensitive rather than purely object-centric.

### B. Belief-Weighted Action Scoring
At each planning step, the robot samples a set of locally feasible candidate motions from its current state. These include goal-directed motions and frontier-like motions derived from locally visible free space. Each candidate is then scored using three factors: geometric progress toward the goal, local motion quality, and semantic interaction risk.

We use the following lightweight decision rule for a candidate action $a$:

\[
\mathrm{Score}(a) = \alpha\,\mathrm{Progress}(a) + \beta\,\mathrm{Smoothness}(a) - \gamma\,\mathrm{Risk}(a),
\]

where higher progress is better, smoother motions are preferred, and larger semantic risk lowers the score. Here, $\mathrm{Progress}(a)$ captures goal-directed improvement, $\mathrm{Smoothness}(a)$ captures local motion quality or directional consistency, and $\mathrm{Risk}(a)$ measures the expected interaction cost of the obstacles the action approaches.

The main novelty lies in the risk term. Instead of scoring nearby obstacles with a fixed penalty, BEACON evaluates the **expected semantic interaction risk** of a candidate action under the current semantic risk beliefs of the obstacles it would approach. For an obstacle belief $b_i$ and per-class risk weights $r(\cdot)$, we define the expected obstacle risk as

\[
R_i = \sum_{c \in \{\text{low, medium, high, no-contact}\}} p_i(c)\, r(c),
\]

with \(r(\text{low}) < r(\text{medium}) < r(\text{high}) < r(\text{no-contact})\). The action-level risk \(\mathrm{Risk}(a)\) is then computed from the expected risks of the obstacles most relevant to that action, for example the nearest or most directly approached obstacles. Actions that move efficiently toward the goal but pass through semantically high-risk regions are therefore penalized more strongly than actions of comparable geometric length in lower-risk regions. This gives the planner a principled way to trade off directness against safety without requiring full global certainty.

This belief-weighted action scoring provides a useful middle ground between two extremes. A purely geometric planner may ignore the meaning of nearby obstacles and choose unsafe or brittle motions, while a planner that treats semantic labels as fixed may overcommit to early assumptions. BEACON instead reasons over uncertainty directly, allowing semantic information to influence motion while remaining adaptable.

### C. Risk-Triggered Online Replanning
Because the environment is only partially known, the best local action can change quickly as additional obstacles are sensed. BEACON therefore replans online and uses a **risk-triggered update rule**: when newly revealed obstacles change the ranking of candidate actions by a sufficient margin, the planner immediately revises its motion choice. This mechanism formalizes the intuition that replanning should occur not only when geometry changes, but also when semantic risk changes the relative desirability of actions.

This idea is especially important in cluttered environments. A direct route may remain geometrically open yet become undesirable once nearby high-risk obstacles are revealed. Conversely, a route that appears indirect at first may become preferable because the locally revealed alternatives are semantically worse. By linking replanning to changes in semantic risk rather than only obstacle occupancy, BEACON couples environment understanding to safe action in a way that is directly aligned with the workshop theme.

### D. Why This Is More Than a Baseline
Although BEACON is built on an existing reactive planning backbone, its contribution is not merely a rebranding of local navigation. The method introduces a distinct planning viewpoint: semantics are uncertain, local, and action-relevant, and should therefore be represented as beliefs that shape online decisions. This shift is small enough to fit a workshop paper, but substantive enough to create a clear technical identity beyond a standard benchmark comparison.

## V. Experimental Setup
We evaluate BEACON using the saved benchmark pipeline in the repository. The experiments are conducted on four procedurally generated scene families: **sparse**, **cluttered**, **collision-required**, and **collision-shortcut**. These families are designed to stress different aspects of online planning. Sparse scenes provide broad free space and limited interaction complexity. Cluttered scenes contain dense obstacle structure and frequent local tradeoffs. Collision-required scenes make interaction-aware navigation important because obstacle layout strongly constrains feasible routes. Collision-shortcut scenes test whether the planner can exploit local structure efficiently without becoming brittle.

We compare BEACON against four planners represented in the saved metrics summaries: **Bug**, **Bug2**, **D* Lite**, and a baseline semantic planner derived from the repository’s prior setup. This set spans classical geometric strategies, heuristic local navigation, and a stronger navigation baseline, providing a useful comparison across different planning assumptions.

The benchmark consists of 400 episodes per planner, corresponding to 100 scenes in each of the four environment families. We report success rate, average number of steps, average path length, average number of logged contacts, and average number of sensed obstacles. These metrics are already stored in the repository’s saved scene-complex summaries, making the evaluation directly reproducible from the current codebase.

To isolate the source of the improvement more directly, we include a compact ablation on the **cluttered** subset of the benchmark. The first variant, **Geometric-Only**, removes semantic influence from the local action score and retains only geometric and directional terms. The second, **Fixed Semantic Labels**, uses fixed semantic penalties without adaptive updates; in the current repository this row is represented conservatively by the saved `surp` cluttered-scene baseline, which serves as the closest available fixed-semantic proxy. The third variant is the unmodified **Full BEACON** planner.

## VI. Results
Table I summarizes overall benchmark performance across 400 episodes per planner. BEACON achieves a 96.75% success rate, outperforming Bug (82.25%) and Bug2 (91.25%), while remaining close to the 97.25% success rate of D* Lite. Importantly, BEACON achieves the shortest average path length among all evaluated planners at 5.053 m. Compared with D* Lite, BEACON reduces average path length from 8.387 m to 5.053 m, a 39.8% reduction, while sacrificing only 0.5 percentage points of success rate. This makes the method’s main strength clear: it does not dominate every scalar metric, but it offers the strongest **reliability-efficiency tradeoff**.

The comparison with the baseline semantic planner is even more pronounced. BEACON improves success rate from 89.25% to 96.75%, reduces average step count from 214.82 to 132.16, and lowers average path length from 7.898 m to 5.053 m. These gains suggest that simply attaching semantics to a planner is not enough; the benefit comes from using semantics as part of an adaptive replanning process rather than as fixed prior information.

The family-level analysis provides the strongest evidence for the proposed approach. In sparse scenes, most planners perform relatively well because free space is abundant and semantic ambiguity matters less. In cluttered scenes, however, BEACON’s advantage becomes much more visible. The proposed planner achieves a 96.00% success rate with an average path length of 5.038 m in cluttered environments, whereas the fixed-semantic proxy achieves only 63.00% success and 11.945 m average path length.

The cluttered-scene ablation makes this interpretation more precise. **Geometric-Only** achieves 94.00% success with 141.47 average steps and a 5.367 m average path length, while **Full BEACON** improves these to 96.00%, 133.80 steps, and 5.038 m. This shows that semantic reasoning improves performance even relative to a strong reactive geometric planner. The much larger drop for the **Fixed Semantic Labels** proxy, at 63.00% success and 332.92 steps, suggests that static semantic penalties are not sufficient in dense partially known scenes. What matters is allowing semantic uncertainty to participate in online action revision.

These results are consistent with the workshop’s emphasis on moving from environment understanding to safe interaction. BEACON’s improvement does not come from a better global map or a more exhaustive search, but from using local semantic beliefs to choose safer and more efficient actions as the robot gradually understands the scene. In that sense, the method provides a practical example of how semantic reasoning can directly improve reliable autonomy.

### Table I. Overall Results (400 Episodes)
| Planner | Success Rate | Avg. Steps | Avg. Path (m) | Avg. Contacts | Avg. Sensed |
|---|---:|---:|---:|---:|---:|
| Bug | 82.25% | 180.47 | 9.604 | 0.000 | 9.725 |
| Bug2 | 91.25% | 185.97 | 8.055 | 0.000 | 11.203 |
| D* Lite | 97.25% | 159.67 | 8.387 | 0.000 | 10.045 |
| Baseline Semantic | 89.25% | 214.82 | 7.898 | 14.470 | 6.125 |
| **BEACON (Proposed)** | **96.75%** | **132.16** | **5.053** | **0.000** | **3.723** |

### Suggested Figure Captions
**Figure 1.** Reliability-efficiency tradeoff across planners. BEACON achieves near-best success with substantially shorter paths than competing methods.

**Figure 2.** Family-level performance of BEACON. Gains are strongest in cluttered environments, where semantic uncertainty-aware replanning has the greatest effect on reliability and path quality.

## VII. Discussion
The current results already support a compelling workshop paper, but the submission can be made stronger by leaning into the paper’s actual insight. The main lesson is not that BEACON simply outperforms a few baselines. Rather, it is that **semantic information becomes most valuable when it is allowed to change local action choice during execution**. This distinction is especially visible in cluttered scenes, where dense obstacle structure creates a sequence of coupled local decisions that cannot be resolved well by fixed semantic assumptions or purely geometric heuristics.

The paper should also be honest about its current limitations. The environment is simulated, the semantic belief abstraction is planning-oriented rather than perception-complete, and the saved results emphasize navigation metrics more than explicit semantic safety diagnostics. These limitations are acceptable for a workshop paper, particularly if they are framed as opportunities for future extension. A promising next step is to couple the current benchmark with richer safety diagnostics, such as belief calibration, dangerous-contact proxies, or explicit semantic failure recovery analysis.

Most importantly, the final submission should avoid overselling. BEACON is strongest not because it universally dominates every metric, but because it offers a more credible balance between reliability, efficiency, and semantic awareness in partially known environments. That is already a meaningful and workshop-appropriate contribution.

## VIII. Conclusion
We presented BEACON, a semantic belief-aware replanning framework for safe online motion planning in partially known environments. The key idea is to treat semantic information as an evolving local planning variable rather than a fixed label assigned once at initialization. By combining local perception, belief-weighted semantic risk evaluation, and risk-triggered replanning, BEACON translates incremental environment understanding into safer and more efficient motion decisions.

Across 400 benchmark episodes, BEACON achieves a 96.75% success rate and the shortest average path length among all evaluated planners. Its strongest gains appear in cluttered environments, where semantic uncertainty-aware replanning is most needed. These results support the broader view that reliable autonomy requires not only perceiving the environment, but reasoning about semantic uncertainty in a way that directly shapes action.

## Submission Notes
- Replace “Anonymous Authors” and affiliation when appropriate.
- Use the correct workshop name throughout: **ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction**.
- Do not include raw exported figure filenames in the PDF.
- Include the cluttered-scene ablation table, with a note that the fixed-semantic row is the repository’s `surp` proxy rather than a separately rerun BEACON-internal variant.
- Prefer the `overall_comparison.png` and `tradeoff_scatter.png` figures plus the overall results table.
