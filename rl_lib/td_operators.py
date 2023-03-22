import jax
import jax.numpy as jnp
import rlax

def bellman_uncertainty():
    pass

def soft_watkins(q_s, a_s, r_s, discount_s, target_qs, kappa, _lambda):
    """
    Assumes;
        q_s is a list from t=1 to t=T+1
        a_s is a list from t=0 to t=T
        r_s is a list from t=0 to t=T
    """

    def lambda_at_t(q_t, a_t, kappa, _lambda):
        """
        In the `off-policy` case, the mixing factor is a function of state, and
        different definitions of `lambda` implement different off-policy corrections:
            Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
            V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

        We use 
            λₜ = λ \mathbb E_{a\sim \pi(\cdot|x_t)} [ \mathbb I_{Q(x_t, a_t) \ge Q(x_t, a) - \kappa \mid Q(x_i, a)\mid}]

        From https://openreview.net/pdf?id=JtC6yOHRoJJ
        """
        id_fn = jnp.where(q_t[a_t] >= q_t - kappa*jnp.abs(q_t), 1, 0)
        pi = rlax.softmax().probs(q_t)
        lambda_t =  _lambda * jnp.inner(pi, id_fn)
        return lambda_t

    lambda_at_t = jax.vmap(lambda_at_t, in_axes=(0, 0, None, None))

    target_as = jnp.concatenate([a_s[1:], jnp.argmax(target_qs[-1], axis=-1, keepdims=True)], axis=0)
    lambda_t = lambda_at_t(target_qs, target_as, kappa, _lambda)

    return rlax.q_lambda(
        q_tm1=q_s,
        a_tm1=a_s,
        r_t=r_s,
        discount_t=discount_s,
        q_t=target_qs,
        lambda_=lambda_t
    )
