import numpy as np
import torch
import torch.nn.functional as F


def compute_loss(
    theta,
    data,
    alpha,
    beta,
    pi_theta,
    pi_imp,
    pi_theta_prime,
    b,
    Gamma_theta_prime,
    p_theta,
    Q_theta,
    eta,
):
    r"""
    Computes the loss L^Q(theta).

    Formula:
    L^Q(\theta) = E_D [
        (1-\alpha) D_{KL}[\pi_{\text{imp}}, \pi_\theta | s_t, \tau, \tilde{\pi} = \pi_{\theta'}]
        + \alpha D_{KL}[b, \pi_\theta | s_t, \tau]
        + \beta D_{KL} [\Gamma_{\theta'}(q | s_t, a_t, \tau), p_\theta(q | s_t, a_t, \tau)]
        - E_D [(1-\alpha) E_{a' \sim \pi_{\theta'}} [w(a', s_t, \tau) \log \pi_\theta(a' | s_t, \tau)]
        + \alpha \log \pi_\theta(a_t | s_t, \tau)
        + \beta E_{q \sim \Gamma_{\theta'}} [\log p_\theta(q | s_t, a_t, \tau)]
        + K_H]

    where w(a', s_t, \tau) = \frac{\exp(Q_\theta(s_t, a', \tau) / \eta)}{E_{a' \sim \pi_{\theta'}} [\exp(Q_\theta(s_t, a', \tau) / \eta)]}
          K_H is a constant entropy related offset independent of \theta.


    Parameters:
    - theta: Parameters of the policy.
    - data: Dataset D containing (s_t, a_t, s_{t+1}, tau) samples.
    - alpha: Scalar weight for the first KL divergence term.
    - beta: Scalar weight for the second KL divergence term.
    - pi_theta: Current policy parameterized by theta.
    - pi_imp: Importance sampling policy.
    - pi_theta_prime: Behavior policy parameterized by theta_prime.
    - b: Baseline policy.
    - Gamma_theta_prime: Distribution parameterized by theta_prime.
    - p_theta: Distribution parameterized by theta.
    - Q_theta: Q-value function parameterized by theta.
    - eta: Temperature parameter.

    Returns:
    - Loss value.
    """
    KH = compute_KH()  # Assuming KH is computed elsewhere

    loss = 0
    for s_t, a_t, s_t1, tau in data:
        # Term 1: (1 - alpha) * D_KL[pi_imp, pi_theta | s_t, tau, tilde_pi = pi_theta_prime]
        term1 = (1 - alpha) * kl_divergence(pi_imp, pi_theta, s_t, tau, pi_theta_prime)

        # Term 2: alpha * D_KL[b, pi_theta | s_t, tau]
        term2 = alpha * kl_divergence(b, pi_theta, s_t, tau)

        # Term 3: beta * D_KL[Gamma_theta_prime(q | s_t, a_t, tau), p_theta(q | s_t, a_t, tau)]
        term3 = beta * kl_divergence(Gamma_theta_prime, p_theta, s_t, a_t, tau)

        # Term 4: -(1 - alpha) * E_{a' ~ pi_theta_prime} [w(a', s_t, tau) log pi_theta(a' | s_t, tau)]
        a_prime_samples = sample_from_policy(pi_theta_prime, s_t, tau)
        w_values = compute_w(a_prime_samples, s_t, tau, Q_theta, eta)
        log_pi_values = np.log(pi_theta(a_prime_samples, s_t, tau))
        term4 = -(1 - alpha) * np.mean(w_values * log_pi_values)

        # Term 5: alpha * log pi_theta(a_t | s_t, tau)
        term5 = alpha * np.log(pi_theta(a_t, s_t, tau))

        # Term 6: beta * E_{q ~ Gamma_theta_prime} [log p_theta(q | s_t, a_t, tau)]
        q_samples = sample_from_distribution(Gamma_theta_prime, s_t, a_t, tau)
        log_p_values = np.log(p_theta(q_samples, s_t, a_t, tau))
        term6 = beta * np.mean(log_p_values)

        loss += term1 + term2 + term3 + term4 + term5 + term6 + KH

    return loss / len(data)


def compute_advantage(pred, eta):
    """Calculate the weighting function w(a', s_t, tau).

    from PAC: https://arxiv.org/pdf/2402.05546
    w(a′,st,τ) = exp(Qθ (st,a′,τ)/η) / E_a′∼πθ′ [exp(Qθ (st,a′,τ)/η)]

    Parameters:
    pred: The current value quantiile predictions.
    eta (float): The temperature parameter.

    Returns:
    weights w(a', s_t, tau) for each action.
    """

    # dont backprop on qval here
    qval = pred["value"]["improve"].mean(-1).clone().detach()
    qval = torch.exp(qval / eta)
    # double check that the mean is over all action proposals
    # dim=2? since bs,seq,nact,nquant
    return qval / qval.mean(2).unsqueeze(-1)


def compute_policy_improvement(log_probs, weights):
    """Compute the policy improvement loss term.

    tensors are (bs, seq, nact)

    Returns:
    torch.Tensor: The computed weighted log-probability term.
    """

    weights = weights.unsqueeze(-1).expand(-1, -1, -1, log_probs.shape[-1])

    # trying something new
    weights = F.log_softmax(weights, dim=2)
    log_probs = F.softmax(log_probs, dim=2)
    loss = F.kl_div(weights, log_probs, log_target=True)
    return loss # positive because we want to minimize the kl divergence

    loss = torch.mean(weights * log_probs, dim=2)
    # its negative because we want to maximize the log prob
    return -torch.mean(loss)


def main():

    # Initialize the parameters
    theta = np.random.rand(10)
    data = np.random.rand(10, 4)
    alpha = 0.5
    beta = 0.5
    pi_theta = np.random.rand(10)
    pi_imp = np.random.rand(10)
    pi_theta_prime = np.random.rand(10)
    b = np.random.rand(10)
    Gamma_theta_prime = np.random.rand(10)
    p_theta = np.random.rand(10)
    Q_theta = np.random.rand(10)
    eta = 0.5
    log_probs = np.random.rand(10, 4, 10)

    """ TODO
    value function is not training right.
    it needs to predict value of the BC actions not of its own actions. (for training)
    at inference time, it will predict its own actions.
    """

    w = compute_advantage(pred, eta)
    loss = compute_policy_improvement(log_probs, w, alpha)

    # Example usage
    states = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Example states
    tau = np.array(
        [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )  # Example context/trajectory information
    actions = np.array(
        [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]]
    )  # Example actions for each state
    weights = np.array(
        [[0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]
    )  # Example weights for each action
    alpha = 0.1  # Example hyperparameter
    weighted_log_prob_term = compute_weighted_log_prob_term(
        states, tau, actions, weights, pi_theta_log_prob, alpha
    )
    print(weighted_log_prob_term)

    loss = compute_loss(
        theta,
        data,
        alpha,
        beta,
        pi_theta,
        pi_imp,
        pi_theta_prime,
        b,
        Gamma_theta_prime,
        p_theta,
        Q_theta,
        eta,
    )


if __name__ == "__main__":
    main()
