"""napkin-returns: which rung of the policy-gradient ladder is load-bearing?

Every course teaches the same staircase: REINFORCE, then add a learned baseline,
then GAE, then PPO's clipped surrogate. Each step "helps". Almost nobody says
which step does the lifting, on what kind of reward, or how many seeds you need
before the ranking you publish is even real.

The ladder here is config flags on one shared implementation, so each rung
changes exactly one thing:

    reinforce   baseline=False  lam=1.0   1 epoch            (MC return-to-go)
    baseline    baseline=True   lam=1.0   1 epoch            (+ learned V(s))
    gae         baseline=True   lam=0.95  1 epoch            (+ bootstrapped adv)
    gaev4       baseline=True   lam=0.95  1 pi / 4 v epochs  (diagnostic: fit V harder)
    reuse       baseline=True   lam=0.95  4 epochs, no clip  (data reuse alone)
    ppo         baseline=True   lam=0.95  4 epochs, clip 0.2 (+ trust region)

Crossed with two reward regimes on the same physics: CartPole-v1 as-is (dense,
+1 per step) and a delayed wrapper that pays the whole episode return at the
final step, so credit assignment has something to fail at. 20 seeds per cell,
matched env steps, matched everything else.

Deliberate design choices, stated up front:
  * Advantages are batch-normalized in ALL methods, including REINFORCE. Without
    it no shared learning rate is fair (MC returns are ~100x the scale of GAE
    advantages). This hands REINFORCE the cheapest baseline for free, so the
    "+baseline" rung measures what a *state-dependent* value adds beyond a batch
    constant -- the honest version of the question.
  * No entropy bonus, no grad clipping, no advantage clipping anywhere. Those
    are rungs for another day; adding them everywhere would blur these four.
  * Value-less methods bootstrap 0 at horizon cuts and truncations (there is no
    V to bootstrap with). That bias is part of what "no critic" costs, not a
    confound.

Usage:
    PYTHONPATH=.deps python3.10 napkin_returns.py selfcheck
    PYTHONPATH=.deps python3.10 napkin_returns.py train --method ppo --env dense --seed 0
    PYTHONPATH=.deps python3.10 napkin_returns.py sweep            # 2x4x20 runs
    PYTHONPATH=.deps python3.10 napkin_returns.py plot             # IQM/CI/ranking charts
    PYTHONPATH=.deps python3.10 napkin_returns.py gif              # untrained vs trained
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym

OUT = Path(__file__).parent / "out"

GAMMA = 0.99
LAM = 0.95
LR_PI, LR_V = 3e-4, 1e-3
NENVS, HORIZON = 8, 256          # 2048 env steps per update
TOTAL_STEPS = 120_000
CLIP, EPOCHS, MINIBATCH = 0.2, 4, 512
HID = 64
SEEDS = 20

METHODS = {
    "reinforce": dict(baseline=False, lam=1.0, ppo=False, clip=None),
    "baseline": dict(baseline=True, lam=1.0, ppo=False, clip=None),
    "gae": dict(baseline=True, lam=LAM, ppo=False, clip=None),
    "gaev4": dict(baseline=True, lam=LAM, ppo=False, clip=None, v_epochs=4),
    "reuse": dict(baseline=True, lam=LAM, ppo=True, clip=None),   # 4 epochs, NO clip
    "ppo": dict(baseline=True, lam=LAM, ppo=True, clip=CLIP),
}
ENV_KINDS = ("dense", "delayed")


# ---------------------------------------------------------------- environments

class Delayed(gym.Wrapper):
    """Accumulate reward, pay it all at the final step (terminal or truncated)."""

    def reset(self, **kw):
        self.acc = 0.0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.acc += r
        out = self.acc if (term or trunc) else 0.0
        return obs, out, term, trunc, info


class Bandit(gym.Env):
    """One-step, one-state: reward 1 for action 0. The smallest MDP that can
    still catch a sign error, a broken terminal mask, or a dead optimizer."""
    observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, **kw):
        return np.zeros(4, np.float32), {}

    def step(self, action):
        return np.zeros(4, np.float32), float(action == 0), True, False, {}


def make_env(kind):
    if kind == "bandit":
        return Bandit()
    env = gym.make("CartPole-v1")
    return Delayed(env) if kind == "delayed" else env


# ---------------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(4, HID), nn.Tanh(),
            nn.Linear(HID, HID), nn.Tanh(),
            nn.Linear(HID, out_dim),
        )

    def forward(self, x):
        return self.f(x)


# -------------------------------------------------------------------- rollout

class Runner:
    """N sequential gym envs stepped in lockstep, manual resets -- no dependence
    on vector-env autoreset semantics, which changed meaning across gym versions."""

    def __init__(self, kind, seed, n=NENVS):
        self.envs = [make_env(kind) for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed * 1000 + i)[0]
                             for i, e in enumerate(self.envs)])
        self.n = n
        self.ep_ret = np.zeros(n)
        self.completed = []          # returns of episodes finished during collect()

    def collect(self, policy, T):
        n = self.n
        obs = np.zeros((T, n, 4), np.float32)
        boot_obs = np.zeros((T, n, 4), np.float32)   # final obs where truncated
        act = np.zeros((T, n), np.int64)
        logp = np.zeros((T, n), np.float32)
        rew = np.zeros((T, n), np.float32)
        term = np.zeros((T, n), bool)
        trunc = np.zeros((T, n), bool)
        for t in range(T):
            obs[t] = self.obs
            with torch.no_grad():
                dist = Categorical(logits=policy(torch.as_tensor(self.obs)))
                a = dist.sample()
                logp[t] = dist.log_prob(a).numpy()
            act[t] = a.numpy()
            for i, e in enumerate(self.envs):
                o2, r, te, tr, _ = e.step(act[t, i])
                rew[t, i], term[t, i], trunc[t, i] = r, te, tr
                self.ep_ret[i] += r
                if te or tr:
                    boot_obs[t, i] = o2
                    self.completed.append(self.ep_ret[i])
                    self.ep_ret[i] = 0.0
                    o2 = e.reset()[0]
                self.obs[i] = o2
        return dict(obs=obs, boot_obs=boot_obs, act=act, logp=logp,
                    rew=rew, term=term, trunc=trunc, last_obs=self.obs.copy())

    def pop_completed(self):
        out, self.completed = self.completed, []
        return out


# ------------------------------------------------------------------------ GAE

def gae(rew, term, trunc, val, boot_val, last_val, gamma, lam):
    """Generalized advantage estimation over a [T, N] batch.

    val: V(s_t); boot_val: V(final obs) where truncated (else ignored);
    last_val: V of the observation after the horizon cut.
    lam=1 with val==0 reduces to MC discounted return-to-go (selfcheck asserts it).
    """
    T, N = rew.shape
    adv = np.zeros((T, N), np.float32)
    nextadv = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else val[t + 1]
        nextval = np.where(trunc[t], boot_val[t], nextval)
        nextval = np.where(term[t], 0.0, nextval)
        delta = rew[t] + gamma * nextval - val[t]
        done = term[t] | trunc[t]
        nextadv = delta + gamma * lam * (~done) * nextadv
        adv[t] = nextadv
    return adv


# --------------------------------------------------------------------- update

def update(policy, value_net, opt_pi, opt_v, batch, cfg):
    T, n = batch["rew"].shape
    flat = lambda x: torch.as_tensor(x.reshape(T * n, *x.shape[2:]))
    obs, act, logp_old = flat(batch["obs"]), flat(batch["act"]), flat(batch["logp"])

    if cfg["baseline"]:
        with torch.no_grad():
            val = value_net(obs).squeeze(-1).numpy().reshape(T, n)
            boot_val = value_net(flat(batch["boot_obs"])).squeeze(-1).numpy().reshape(T, n)
            last_val = value_net(torch.as_tensor(batch["last_obs"])).squeeze(-1).numpy()
    else:
        val = boot_val = np.zeros((T, n), np.float32)
        last_val = np.zeros(n, np.float32)

    adv = gae(batch["rew"], batch["term"], batch["trunc"],
              val, boot_val, last_val, GAMMA, cfg["lam"])
    ret = torch.as_tensor((adv + val).reshape(-1))
    adv = torch.as_tensor(adv.reshape(-1))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    if cfg["ppo"]:
        for _ in range(EPOCHS):
            for idx in torch.randperm(T * n).split(MINIBATCH):
                dist = Categorical(logits=policy(obs[idx]))
                ratio = torch.exp(dist.log_prob(act[idx]) - logp_old[idx])
                if cfg["clip"] is None:
                    loss_pi = -(ratio * adv[idx]).mean()
                else:
                    clipped = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"])
                    loss_pi = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
                loss_v = ((value_net(obs[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                opt_v.zero_grad(); loss_v.backward(); opt_v.step()
    else:
        dist = Categorical(logits=policy(obs))
        loss_pi = -(dist.log_prob(act) * adv).mean()
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
        if cfg["baseline"]:
            for _ in range(cfg.get("v_epochs", 1)):
                loss_v = ((value_net(obs).squeeze(-1) - ret) ** 2).mean()
                opt_v.zero_grad(); loss_v.backward(); opt_v.step()


# ---------------------------------------------------------------------- train

def train(method, env_kind, seed, total_steps=TOTAL_STEPS,
          nenvs=NENVS, horizon=HORIZON, quiet=False):
    cfg = METHODS[method]
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy, value_net = MLP(2), MLP(1)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR_PI)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=LR_V)
    runner = Runner(env_kind, seed, nenvs)

    curve, last = [], 0.0
    for upd in range(total_steps // (nenvs * horizon)):
        batch = runner.collect(policy, horizon)
        update(policy, value_net, opt_pi, opt_v, batch, cfg)
        eps = runner.pop_completed()
        last = float(np.mean(eps)) if eps else last
        curve.append((int((upd + 1) * nenvs * horizon), last))
        if not quiet and (upd + 1) % 10 == 0:
            print(f"  {method}/{env_kind} seed {seed}  "
                  f"steps {curve[-1][0]:>7}  return {last:7.1f}")
    return curve, policy


# ---------------------------------------------------------------------- sweep

def run_sweep(total_steps, seeds):
    OUT.joinpath("sweep").mkdir(parents=True, exist_ok=True)
    t0, todo = time.time(), []
    for kind in ENV_KINDS:
        for method in METHODS:
            for seed in range(seeds):
                todo.append((kind, method, seed))
    for k, (kind, method, seed) in enumerate(todo):
        f = OUT / "sweep" / f"{kind}_{method}_{seed}.json"
        if f.exists():
            continue
        curve, _ = train(method, kind, seed, total_steps, quiet=True)
        f.write_text(json.dumps(curve))
        el = time.time() - t0
        print(f"[{k + 1:3}/{len(todo)}] {kind:7} {method:9} seed {seed:2}  "
              f"final {curve[-1][1]:6.1f}  elapsed {el / 60:5.1f} min", flush=True)
    print("sweep done")


def load_sweep():
    runs = {}
    for f in (OUT / "sweep").glob("*.json"):
        kind, method, seed = f.stem.rsplit("_", 2)
        runs.setdefault((kind, method), {})[int(seed)] = json.loads(f.read_text())
    return runs


# ---------------------------------------------------------------- aggregation

def iqm(x, axis=None):
    """Interquartile mean: mean of the middle 50% (rliable's robust aggregate)."""
    x = np.sort(np.asarray(x, np.float64), axis=axis)
    n = x.shape[-1] if axis in (None, -1) else x.shape[axis]
    lo, hi = n // 4, n - n // 4
    sl = [slice(None)] * x.ndim
    sl[-1 if axis in (None, -1) else axis] = slice(lo, hi)
    return x[tuple(sl)].mean(axis=axis)


def bootstrap_ci(x, stat=iqm, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def final_scores(runs, kind):
    """Per-seed mean return over the last 10% of training."""
    out = {}
    for method in METHODS:
        seeds = runs[(kind, method)]
        vals = []
        for s in sorted(seeds):
            curve = np.array(seeds[s])
            tail = curve[curve[:, 0] >= 0.9 * curve[-1, 0], 1]
            vals.append(tail.mean())
        out[method] = np.array(vals)
    return out


def ranking_stability(scores, n_sub=1000, seed=0):
    """For k=1..N random seed subsets: P(best@k == best@N) and, because a tied
    top pair makes 'best' a coin flip at any k, P(top-2 set@k == top-2 set@N)."""
    rng = np.random.default_rng(seed)
    methods = list(scores)
    n = len(scores[methods[0]])
    rank = lambda idx: sorted(methods, key=lambda m: -iqm(scores[m][idx]))
    full = rank(np.arange(n))
    true_best, true_top2 = full[0], frozenset(full[:2])
    ps, ps2 = [], []
    for k in range(1, n + 1):
        hits = hits2 = 0
        for _ in range(n_sub):
            r = rank(rng.choice(n, size=k, replace=False))
            hits += r[0] == true_best
            hits2 += frozenset(r[:2]) == true_top2
        ps.append(hits / n_sub)
        ps2.append(hits2 / n_sub)
    return true_best, ps, ps2


# ----------------------------------------------------------------------- plot

def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_sweep()
    colors = dict(reinforce="#999999", baseline="#4477aa", gae="#ee7733",
                  gaev4="#ddaa33", reuse="#aa3377", ppo="#228833")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, kind in enumerate(ENV_KINDS):
        ax = axes[0, col]
        for method in METHODS:
            seeds = runs[(kind, method)]
            curves = np.array([seeds[s] for s in sorted(seeds)])  # [S, U, 2]
            x = curves[0, :, 0]
            for c in curves:
                ax.plot(c[:, 0], c[:, 1], color=colors[method], alpha=0.12, lw=0.7)
            ax.plot(x, iqm(curves[:, :, 1], axis=0), color=colors[method],
                    lw=2.2, label=method)
        ax.set_title(f"{kind} reward -- 20 seeds each, IQM in bold")
        ax.set_xlabel("env steps"); ax.set_ylabel("episode return")
        ax.legend(loc="upper left", fontsize=8)

    results = {}
    for col, kind in enumerate(ENV_KINDS):
        scores = final_scores(runs, kind)
        best, ps, ps2 = ranking_stability(scores)
        ax = axes[1, 0]
        line, = ax.plot(range(1, len(ps) + 1), ps, marker="o", ms=3,
                        label=f"{kind}: unique best ({best})")
        ax.plot(range(1, len(ps2) + 1), ps2, ls="--", color=line.get_color(),
                label=f"{kind}: top-2 set")
        results[kind] = {
            m: dict(iqm=float(iqm(s)), ci=[float(v) for v in bootstrap_ci(s)],
                    seeds=[float(v) for v in s])
            for m, s in scores.items()
        }
        results[kind]["ranking_stability"] = ps
        results[kind]["top2_stability"] = ps2

        ax = axes[1, 1]
        xs = np.arange(len(METHODS)) + (0.4 if col else 0)
        vals = [results[kind][m]["iqm"] for m in METHODS]
        errs = np.array([[results[kind][m]["iqm"] - results[kind][m]["ci"][0],
                          results[kind][m]["ci"][1] - results[kind][m]["iqm"]]
                         for m in METHODS]).T
        ax.bar(xs, vals, width=0.4, yerr=errs, capsize=3,
               color=[colors[m] for m in METHODS],
               alpha=1.0 if col == 0 else 0.55,
               label=kind)
        ax.set_xticks(np.arange(len(METHODS)) + 0.2, list(METHODS))

    axes[1, 0].set_title("P(best at k seeds == best at 20 seeds)")
    axes[1, 0].set_xlabel("seeds used"); axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].axhline(0.95, color="k", ls=":", lw=0.8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title("final IQM return, 95% bootstrap CI")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.png'} and results.json")


# ------------------------------------------------------------------------ gif

def draw_cartpole(state, label, step, w=320, h=200):
    from PIL import Image, ImageDraw
    x, _, th, _ = state
    img = Image.new("RGB", (w, h), (250, 250, 245))
    d = ImageDraw.Draw(img)
    d.text((10, 8), label, fill=(70, 70, 70))
    d.text((10, 22), f"step {step}", fill=(150, 150, 150))
    d.line([(0, h - 40), (w, h - 40)], fill=(180, 180, 180), width=2)
    cx = w / 2 + x / 2.4 * (w / 2 - 30)
    d.rectangle([cx - 20, h - 52, cx + 20, h - 40], fill=(60, 60, 70))
    tip = (cx + 80 * np.sin(th), h - 52 - 80 * np.cos(th))
    d.line([(cx, h - 52), tip], fill=(200, 60, 40), width=5)
    return img


def rollout_states(policy, seed, greedy=True, max_t=400):
    env = make_env("dense")
    obs, _ = env.reset(seed=seed)
    states = [obs.copy()]
    for _ in range(max_t):
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs[None]))
        a = int(logits.argmax()) if greedy else int(Categorical(logits=logits).sample())
        obs, _, te, tr, _ = env.step(a)
        states.append(obs.copy())
        if te or tr:
            break
    return states


def make_gif():
    from PIL import Image
    print("training ppo/dense for the gif (~1 min)...")
    _, policy = train("ppo", "dense", seed=0, quiet=True)
    before = rollout_states(MLP(2), seed=3)
    after = rollout_states(policy, seed=3)
    n = max(len(before), len(after))
    frames = []
    for t in range(0, n, 2):
        l = draw_cartpole(before[min(t, len(before) - 1)], "untrained",
                          min(t, len(before) - 1))
        r = draw_cartpole(after[min(t, len(after) - 1)], "trained (ppo, 120k steps)",
                          min(t, len(after) - 1))
        img = Image.new("RGB", (650, 220), (250, 250, 245))
        img.paste(l, (0, 20)); img.paste(r, (330, 20))
        frames.append(img)
    frames[0].save(OUT / "cartpole.gif", save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    print(f"wrote {OUT / 'cartpole.gif'}  (left: untrained {len(before) - 1} steps, "
          f"right: trained {len(after) - 1} steps)")


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    rng = np.random.default_rng(0)
    T, N = 40, 3
    rew = rng.normal(size=(T, N)).astype(np.float32)
    val = rng.normal(size=(T, N)).astype(np.float32)
    boot = rng.normal(size=(T, N)).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)
    term = np.zeros((T, N), bool); term[13, 0] = term[29, 2] = True
    trunc = np.zeros((T, N), bool); trunc[21, 1] = True

    # 1. GAE(lam=1, val=0) == MC discounted return-to-go, resetting at dones,
    #    zero bootstrap. Independently derived forward recursion:
    zeros = np.zeros_like(val)
    adv1 = gae(rew, term, trunc, zeros, zeros, np.zeros(N, np.float32), GAMMA, 1.0)
    mc = np.zeros_like(rew)
    for i in range(N):
        acc = 0.0
        for t in reversed(range(T)):
            if term[t, i] or trunc[t, i]:
                acc = 0.0
            mc[t, i] = rew[t, i] + GAMMA * acc
            acc = mc[t, i]
    err = np.abs(adv1 - mc).max()
    assert err < 1e-5, f"GAE(1) != MC return-to-go: {err}"
    print(f"GAE(lam=1, V=0) == MC return-to-go        max err {err:.1e}")

    # 2. GAE(lam=0) == one-step TD residual, with terminal/truncation masking.
    adv0 = gae(rew, term, trunc, val, boot, last, GAMMA, 0.0)
    nextv = np.vstack([val[1:], last[None]])
    nextv = np.where(trunc, boot, nextv)
    nextv = np.where(term, 0.0, nextv)
    td = rew + GAMMA * nextv - val
    err = np.abs(adv0 - td).max()
    assert err < 1e-6, f"GAE(0) != TD residual: {err}"
    print(f"GAE(lam=0) == TD residual                 max err {err:.1e}")

    # 3. Truncation bootstraps V(final obs); termination bootstraps 0.
    assert abs(adv0[21, 1] - (rew[21, 1] + GAMMA * boot[21, 1] - val[21, 1])) < 1e-6
    assert abs(adv0[13, 0] - (rew[13, 0] - val[13, 0])) < 1e-6
    print("truncation uses V(final obs), termination uses 0")

    # 4. PPO surrogate at clip=inf, 1 epoch, full batch has the same gradient
    #    as the vanilla PG loss (ratio == 1 at theta == theta_old).
    torch.manual_seed(0)
    policy = MLP(2)
    obs = torch.randn(64, 4)
    act = torch.randint(0, 2, (64,))
    adv = torch.randn(64)
    dist = Categorical(logits=policy(obs))
    logp_old = dist.log_prob(act).detach()
    pg = -(dist.log_prob(act) * adv).mean()
    g_pg = torch.autograd.grad(pg, list(policy.parameters()))
    dist2 = Categorical(logits=policy(obs))
    ratio = torch.exp(dist2.log_prob(act) - logp_old)
    surr = -torch.min(ratio * adv, torch.clamp(ratio, -1e9, 1e9) * adv).mean()
    g_su = torch.autograd.grad(surr, list(policy.parameters()))
    rel = max((a - b).abs().max().item() / (a.abs().max().item() + 1e-12)
              for a, b in zip(g_pg, g_su))
    assert rel < 1e-5, f"PPO(clip=inf) grad != PG grad: rel {rel}"
    print(f"PPO(clip=inf, 1 epoch) grad == PG grad    rel err {rel:.1e}")

    # 5. Delayed wrapper conserves the episode total and pays only at the end.
    e1, e2 = make_env("dense"), make_env("delayed")
    o1 = e1.reset(seed=7); o2 = e2.reset(seed=7)
    tot1 = tot2 = 0.0; mid = []
    done = False
    while not done:
        a = int(rng.integers(2))
        _, r1, te1, tr1, _ = e1.step(a)
        _, r2, te2, tr2, _ = e2.step(a)
        tot1 += r1; tot2 += r2
        done = te1 or tr1
        if not done:
            mid.append(r2)
    assert tot1 == tot2 and all(m == 0 for m in mid), (tot1, tot2)
    print(f"delayed wrapper: same total ({tot1:.0f}), all of it on the last step")

    # 6. Same seeds -> identical rollouts (determinism, as train() seeds it).
    def det_rollout():
        torch.manual_seed(1)
        return Runner("dense", 5).collect(MLP(2), 32)
    b1, b2 = det_rollout(), det_rollout()
    assert np.array_equal(b1["obs"], b2["obs"]) and np.array_equal(b1["act"], b2["act"])
    print("rollouts are seed-deterministic")

    # 7. Every method must solve the 2-armed bandit (end-to-end machinery test).
    for method in METHODS:
        curve, policy = train(method, "bandit", seed=0, total_steps=8 * 16 * 150,
                              nenvs=8, horizon=16, quiet=True)
        with torch.no_grad():
            p0 = torch.softmax(policy(torch.zeros(1, 4)), -1)[0, 0].item()
        assert p0 > 0.9, f"{method} failed the bandit: P(a=0)={p0:.3f}"
        print(f"{method:9} solves the bandit           P(best arm) {p0:.3f}")

    print("\nall selfchecks passed")


# ------------------------------------------------------------------------ cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    t = sub.add_parser("train")
    t.add_argument("--method", choices=METHODS, default="ppo")
    t.add_argument("--env", choices=ENV_KINDS, default="dense")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s = sub.add_parser("sweep")
    s.add_argument("--steps", type=int, default=TOTAL_STEPS)
    s.add_argument("--seeds", type=int, default=SEEDS)
    sub.add_parser("plot")
    sub.add_parser("gif")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "train":
        t0 = time.time()
        curve, _ = train(args.method, args.env, args.seed, args.steps)
        print(f"final return {curve[-1][1]:.1f}  ({time.time() - t0:.0f}s)")
    elif args.cmd == "sweep":
        run_sweep(args.steps, args.seeds)
    elif args.cmd == "plot":
        make_plots()
    elif args.cmd == "gif":
        make_gif()
