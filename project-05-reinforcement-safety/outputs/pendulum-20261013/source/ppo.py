"""NumPy PPO with analytic gradients and a tanh-transformed Gaussian actor."""
import numpy as np


class Network:
    def __init__(self, inputs, outputs, rng, actor=False):
        self.p = {"w0": rng.normal(size=(inputs, 32))/np.sqrt(inputs), "b0": np.zeros(32),
                  "w1": rng.normal(size=(32, 32))/np.sqrt(32), "b1": np.zeros(32),
                  "w2": rng.normal(size=(32, outputs))*(0.01 if actor else 1)/np.sqrt(32), "b2": np.zeros(outputs)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        a = np.tanh(x@self.p["w0"]+self.p["b0"])
        b = np.tanh(a@self.p["w1"]+self.p["b1"])
        return b@self.p["w2"]+self.p["b2"], (x, a, b)

    def backward(self, cache, dy):
        x, a, b = cache
        g = {"w2": b.T@dy, "b2": dy.sum(axis=0)}
        db = (dy@self.p["w2"].T)*(1-b*b)
        g.update(w1=a.T@db, b1=db.sum(axis=0))
        da = (db@self.p["w1"].T)*(1-a*a)
        g.update(w0=x.T@da, b0=da.sum(axis=0))
        return g

    def update(self, grads, lr):
        self.t += 1
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        for k, gradient in grads.items():
            g = gradient*min(1, 0.5/max(norm, 1e-12))
            self.m[k] = .9*self.m[k]+.1*g
            self.v[k] = .999*self.v[k]+.001*g*g
            self.p[k] -= lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)


def log_probability(z, mean, log_std):
    """Density of a=tanh(z); action scaling is handled by the environment."""
    normal = (-.5*((z-mean)*np.exp(-log_std))**2-log_std-.5*np.log(2*np.pi)).sum(axis=-1)
    log_jacobian = 2*(np.log(2)-z-np.logaddexp(0, -2*z))
    return normal-log_jacobian.sum(axis=-1)


def gae(rewards, values, next_values, terminated, truncated, gamma=.97, lam=.95):
    """Bootstrap on truncation; stop the GAE recurrence across every reset."""
    delta = rewards+gamma*next_values*(~terminated)-values
    advantages = np.zeros_like(delta)
    tail = np.zeros(delta.shape[1:])
    for t in range(len(delta)-1, -1, -1):
        tail = delta[t]+gamma*lam*(~(terminated[t]|truncated[t]))*tail
        advantages[t] = tail
    return advantages, advantages+values


class Agent:
    def __init__(self, obs_dim, action_dim, seed=0):
        self.rng = np.random.default_rng(seed)
        self.actor = Network(obs_dim, action_dim, self.rng, actor=True)
        self.critic = Network(obs_dim, 1, self.rng)
        # Learnable log std uses the same Adam implementation.
        self.actor.p["log_std"] = np.full(action_dim, -.5)
        self.actor.m["log_std"] = np.zeros(action_dim)
        self.actor.v["log_std"] = np.zeros(action_dim)

    def sample(self, obs, deterministic=False):
        mean, _ = self.actor.forward(obs)
        z = mean if deterministic else mean+np.exp(self.actor.p["log_std"])*self.rng.normal(size=mean.shape)
        return np.tanh(z), z, log_probability(z, mean, self.actor.p["log_std"])

    def value(self, obs):
        return self.critic.forward(obs)[0][..., 0]

    def actor_loss_grad(self, obs, z, old_logp, advantages, clip=.2, entropy_coef=0.0):
        mean, cache = self.actor.forward(obs)
        ls = self.actor.p["log_std"]
        lp = log_probability(z, mean, ls)
        ratio = np.exp(lp-old_logp)
        clipped = np.clip(ratio, 1-clip, 1+clip)
        entropy = float(np.sum(ls+.5*np.log(2*np.pi*np.e)))
        loss = -float(np.minimum(ratio*advantages, clipped*advantages).mean())-entropy_coef*entropy
        active = ((advantages >= 0)&(ratio <= 1+clip))|((advantages < 0)&(ratio >= 1-clip))
        derivative = -active*advantages*ratio/len(obs)
        invvar = np.exp(-2*ls)
        gradients = self.actor.backward(cache, derivative[:, None]*(z-mean)*invvar)
        gradients["log_std"] = (derivative[:, None]*(((z-mean)**2)*invvar-1)).sum(axis=0)-entropy_coef
        diagnostics = {"actor_loss": loss, "approx_kl": float(((ratio-1)-(lp-old_logp)).mean()),
                       "clip_fraction": float(np.mean(np.abs(ratio-1)>clip)), "latent_gaussian_entropy": entropy}
        return loss, gradients, diagnostics

    def optimize(self, obs, z, old_logp, advantages, returns, epochs=6, lr=3e-4):
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        logs = []
        stopped = False
        for _ in range(epochs):
            ids = self.rng.permutation(len(obs))
            for start in range(0, len(ids), 256):
                b = ids[start:start+256]
                _, gradient, diag = self.actor_loss_grad(obs[b], z[b], old_logp[b], advantages[b])
                if diag["approx_kl"] > .03:
                    stopped = True
                    break
                self.actor.update(gradient, lr)
                self.actor.p["log_std"] = np.clip(self.actor.p["log_std"], -2, .5)
                prediction, cache = self.critic.forward(obs[b])
                error = prediction[:, 0]-returns[b]
                self.critic.update(self.critic.backward(cache, error[:, None]/len(b)), lr)
                logs.append({**diag, "value_loss": float(np.mean(error**2)/2)})
            if stopped:
                break
        # Include the guard even when the first minibatch is rejected.
        result = {k: float(np.mean([r[k] for r in logs])) for k in logs[0]} if logs else {**diag, "value_loss": float(np.mean((self.value(obs)-returns)**2)/2)}
        result["kl_early_stop"] = stopped
        return result

    def save(self, path):
        np.savez_compressed(path, **{f"actor_{k}": v for k, v in self.actor.p.items()},
                            **{f"critic_{k}": v for k, v in self.critic.p.items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            agent = cls(data["actor_w0"].shape[0], data["actor_w2"].shape[1])
            agent.actor.p = {k: data[f"actor_{k}"].copy() for k in agent.actor.p}
            agent.critic.p = {k: data[f"critic_{k}"].copy() for k in agent.critic.p}
        return agent
