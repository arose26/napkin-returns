# Insights from building napkin-returns

Written in the order I hit them, same convention as the earlier napkin repos: most of these
showed up as a failed assert or a number that disagreed with a formula, not from rereading code.

## 1. A determinism test can fail because of *your test's* RNG plumbing, not the code's

The selfcheck "same seed → identical rollouts" assert failed on first run. The rollout code was
fine: the two rollouts shared the **global torch RNG stream**, so the first `collect()` advanced
it and the second sampled different actions from the same policy. The env seeding was airtight;
the *test* wasn't reproducing the thing training actually does (seed everything, then run).

**Takeaway:** a determinism test must recreate the full seeding path of the code it certifies —
seed, build, run — not share ambient RNG state between its two arms. The failure looked exactly
like nondeterministic env physics and was nothing of the sort.

## 2. When the weakest method fails a convergence check, raise the budget, not the bar

REINFORCE hit P(best arm)=0.807 on the bandit selfcheck against a 0.9 threshold — learning,
just slower than the other three rungs (which is the entire premise of the repo). The tempting
fix is threshold 0.8. The right fix is more updates: at 150 updates all four methods clear 0.99,
and the assert still catches a sign error or dead optimizer instantly.

**Takeaway** (the RL twin of napkin-diffusion insight #9): loosening a threshold silently
destroys a test's power. If the quantity being tested is *convergence*, the knob to turn is
compute, and the threshold should stay where failure is unambiguous.

## 3. Batch-normalizing advantages is not a neutral implementation detail — it is a rung of the ladder

Placing "normalize advantages" in the shared code path means vanilla REINFORCE silently gets a
batch-mean baseline. Most public "REINFORCE vs PPO" comparisons do this without noticing that
their REINFORCE is already half a rung up the ladder. We kept the normalization (without it no
shared learning rate is fair across methods whose raw advantage scales differ by ~100×) and
rewrote the question it answers: the `+baseline` rung here measures what a *state-dependent*
value adds beyond a batch constant.

**Takeaway:** in an ablation, every line of shared code is part of the experimental design.
The alternative — per-method learning rates — would have confounded the ladder with a tuning
contest.

## 4. When a result surprises you, split the bundle before shipping the headline

The first sweep said "PPO wins 6× on dense". But PPO is two ideas — 4 epochs of data reuse, and
a ratio clip to make the reuse safe. Shipping "PPO wins" would have been true and useless. One
extra arm (`reuse`: 4 epochs, no clip) showed the clip contributed **nothing measurable in any
cell of this experiment**: the entire 6× is data reuse. Then the same knife applied to `reuse`
itself (`gaev4`: reuse only the value updates) killed our mechanism story for the delayed-reward
rescue — value-side reuse recovers a quarter of the collapse, not most of it.

**Takeaway:** a named method is a bundle of decisions. An ablation that stops at the bundle
boundary measures marketing, not mechanism. Each split here cost 40 runs and ~8 minutes.

## 5. Bootstrapped advantages can be strictly worse than no critic

GAE(λ=0.95) on delayed reward scored 14.8 where plain REINFORCE scored 40.0 — and its learning
curve *decreases*. The terminal lump's credit reaches an action k steps back attenuated by
(γλ)^k, so early actions see advantages that are almost entirely value-estimation error; batch
normalization then rescales that noise to unit magnitude, and the policy walks confidently away
from where it started. "More sophisticated estimator" + "wrong regime" = active harm, not
graceful degradation.

**Takeaway:** GAE's λ trades variance against *bias from an inaccurate V*. When reward structure
makes V hard to learn (a single terminal payout), that bias is the whole signal. Check what your
advantage estimator degrades into when its inputs are garbage, because that is what it will be
computing for the first N updates of every run.

## 6. A ranking-stability metric can diagnose the question, not just the seed budget

P(best@k == best@20) on dense saturated at ~0.5 for every k. First reading: "even 20 seeds
aren't enough". Actual reading: `reuse` and `ppo` are tied, so "the best method" is a coin flip
at any budget — the experiment's question had no answer, and the metric was pointing at the
question. The top-2 *set*, by contrast, is stable from a single seed. On delayed, where a unique
winner exists, naming it reliably takes ~15 seeds.

**Takeaway** (the series' evaluation rule from here on): before asking "how many seeds until the
ranking stabilizes", ask "is the thing I'm ranking actually distinct". Report whichever object is
stable — a winner, a tied pair, a set — rather than forcing a total order onto noise. And note
the k=N point of any subset-stability curve is 1.0 by construction; it carries no information.
