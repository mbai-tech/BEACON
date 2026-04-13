# Methods

## Risk-Aware Online Motion Planning with Local Sensing and Controlled Contact

We study a point-robot motion planning problem in an initially unknown cluttered environment. The robot knows its start and goal positions, but it does not know obstacle locations in advance. Instead, it senses obstacles only within a local radius \(R_s\). The robot moves greedily toward the goal whenever possible, but it must stop at a safety distance \(\epsilon\) before colliding with an unseen obstacle. Once obstacles are sensed, the robot decides whether to avoid them or to push them in a controlled way to open a traversable corridor.

Let the robot position be \(p \in \mathbb{R}^2\) and the goal be \(g \in \mathbb{R}^2\). The nominal goal-seeking direction is

\[
\hat e_g = \frac{g - p}{\lVert g - p \rVert},
\]

so the direct motion policy is a bounded one-step version of

\[
u_{\text{goal}} = v_{\max} \hat e_g.
\]

At each control cycle, the robot first performs a local perception update. Obstacles whose body clearance is within sensing range are marked as observed. In addition, radial sensing rays are cast around the robot to approximate nearby free space and frontier structure. Rays that reach the sensing horizon without hitting an obstacle are treated as local frontiers, which indicate the boundary between known free space and still-unexplored space. This yields a local, online approximation of occupancy and blocking structure without requiring a global map.

To enforce safe first contact, the robot uses an \(\epsilon\)-stop rule. For each obstacle \(O_i\), let \(d_i(p)\) denote the clearance between the robot body and obstacle \(O_i\). The robot is allowed to continue direct motion only while

\[
\min_i d_i(p) > \epsilon.
\]

If a motion step would cause the robot to enter the \(\epsilon\)-boundary of an unseen obstacle, the step is truncated and the robot pauses to classify the local situation.

Once obstacles are observed, the planner partitions them into candidate avoid and push sets. In the current project version, all generated obstacles are semantically pushable, but the planner still evaluates whether pushing is a good choice. Each sensed obstacle is assigned a semantic safety score using simple feature proxies for mass, friction, fragility, semantic risk, and confidence:

\[
S_i = 0.10\,m_i + 0.15\,\mu_i + 0.45\,f_i + 0.40\,c_i - 0.25\,q_i,
\]

which is then converted to a push-safety probability

\[
P_{\text{safe}}(i) = \sigma(-S_i) = \frac{1}{1 + e^{S_i}}.
\]

Higher \(P_{\text{safe}}\) indicates a better push candidate.

The planner then evaluates two local trajectory branches in parallel. The avoidance branch computes a short sidestep or escape maneuver around nearby obstacles. Its cost is

\[
J_{\text{avoid}} = 1.1\,L_{\text{step}} + 0.7\,D_{\text{remain}} + 0.25\,U,
\]

where \(L_{\text{step}}\) is the immediate step length, \(D_{\text{remain}}\) is the remaining distance to the goal after taking that step, and \(U\) is a local uncertainty term. The push branch estimates whether pushing a nearby obstacle would create a corridor. Its cost is

\[
J_{\text{push}} = 0.6\,T + 0.5\,E + 1.2\,R - 1.4\,G,
\]

where \(T\) is a time term, \(E\) is push effort, \(R\) is semantic/contact risk, and \(G\) is corridor gain, defined as the increase in robot-to-obstacle clearance produced by the predicted push. Pushing is favored when it is safe, low-effort, and creates useful free space.

When the planner selects a push action, the obstacle is moved in a direction aligned with the robot’s intended motion while still enforcing outward contact geometry. If \(\hat e_{\text{pref}}\) is the robot’s preferred travel direction and \(\hat e_{\text{out}}\) points from the robot toward the obstacle center, the push direction is

\[
\hat d_{\text{push}} = \text{normalize}(0.35\,\hat e_{\text{pref}} + 0.65\,\hat e_{\text{out}}).
\]

This prevents visually implausible motion in which the obstacle appears to move back toward the robot. If pushing causes one obstacle to contact another, the resulting chain motion is modeled with attenuation:

\[
\Delta_k = \Delta_0 \cdot \gamma^{k},
\]

where \(\Delta_0\) is the lead obstacle displacement, \(k\) is the contact-chain level, and \(\gamma \in (0,1)\) is a fixed attenuation factor. This yields a simple quasi-static approximation to force propagation with energy loss.

Finally, the planner reconciles the avoid and push branches with a safety-gated decision rule and executes only one short step before replanning. If progress toward the goal stalls, measured by insufficient decrease in \(\|g-p\|\) over a rolling window, the planner escalates through recovery modes: direct push of a blocking obstacle, deeper path backtracking, and a local RRT-style escape search over currently sensed free space. This repeated sense-decide-act loop makes the method fully online and suitable for unknown environments where obstacle interaction may be necessary to reach the goal.

---

## Algorithm 1: Risk-Aware Online Avoid-or-Push Planning

```text
Algorithm 1  Risk-Aware Online Avoid-or-Push Planning
Input:
    scene with start s and goal g
    sensing range R_s
    safety distance epsilon
    step size delta
    push distance delta_push
Output:
    robot path P

1:  Initialize robot position p <- s
2:  Initialize path P <- [p]
3:  Initialize local memory M_bad <- empty
4:  Initialize push history H_push <- empty
5:  while ||g - p|| > delta do
6:      Perform local perception update within radius R_s
7:      Reveal obstacles entering sensing range
8:      Cast radial sensing rays around p
9:      Identify local frontier points
10:     Update local occupancy/blocking estimates
11:
12:     Compute goal direction e_g <- normalize(g - p)
13:     Find nearest sensed obstacle O*
14:
15:     if no sensed obstacle violates epsilon-boundary then
16:         p_next <- safe direct step toward goal
17:         p <- p_next
18:         Append p to P
19:         continue
20:     end if
21:
22:     Stop at epsilon boundary
23:     Classify sensed obstacles into avoid set A and push set C
24:
25:     Compute avoid trajectory T_a:
26:         generate sidestep / escape candidates
27:         score them by progress, clearance, and step length
28:         smooth best local step
29:
30:     Compute push trajectory T_p for push candidates:
31:         estimate obstacle safety probability P_safe
32:         estimate feasible push distance
33:         estimate corridor gain after push
34:         score push branch by time, effort, risk, and gain
35:
36:     Reconcile T_a and T_p:
37:         reject push if safety margin too small
38:         compare normalized avoid and push branch scores
39:         choose lower-cost / safer branch
40:
41:     if robot is stalled over recent time window then
42:         Record bad local direction in M_bad
43:         if a blocking pushable obstacle is available then
44:             execute push step
45:         else if deeper backtrack is available then
46:             move backward along previous path
47:         else
48:             run local RRT-style escape over sensed free space
49:         end if
50:     else
51:         if push branch selected then
52:             move obstacle in push direction
53:             propagate any chain contact with attenuation
54:             update H_push
55:             move robot one step forward
56:         else
57:             move robot one step along avoid trajectory
58:         end if
59:     end if
60:
61:     Verify outcome and append updated position to P
62: end while
63: return P
```
