"""
CartPole-v1 : Value Function Approximation lineaire, SARSA(0) vs Q-learning.

Reprend directement le formalisme du fichier 02 (Function Approximation) :

    Q_hat(s, a; W) = w_a . x(s)        (une ligne de poids W[a] par action)

    SARSA   : delta = r + gamma * Q_hat(s', a') - Q_hat(s, a)   (a' = action reellement tiree epsilon-greedy)
    Q-learn : delta = r + gamma * max_a' Q_hat(s', a') - Q_hat(s, a)

    W[a] <- W[a] + alpha * delta * x(s)
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Features x(s) : etat brut normalise + biais
# ---------------------------------------------------------------------------
def make_features(obs):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    x = np.array([
        cart_pos / 4.8,
        np.clip(cart_vel, -4, 4) / 4.0,
        pole_angle / 0.418,
        np.clip(pole_vel, -4, 4) / 4.0,
        1.0,  # biais
    ])
    return x


def q_hat(x, W):
    """Retourne le vecteur des Q-valeurs pour toutes les actions : W @ x, shape (n_actions,)."""
    return W @ x


def epsilon_greedy(x, W, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    q_values = q_hat(x, W)
    return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Boucle d'entrainement generique (methode = "sarsa" ou "qlearning")
# ---------------------------------------------------------------------------
def train(method, n_episodes=800, alpha=0.01, gamma=0.99,
          eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=0):

    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0] + 1

    rng = np.random.default_rng(seed)
    W = np.zeros((n_actions, n_features))

    epsilon = eps_start
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        x = make_features(obs)
        a = epsilon_greedy(x, W, epsilon, n_actions, rng)

        total_reward = 0.0
        done = False

        while not done:
            obs_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            x_next = make_features(obs_next)
            total_reward += r

            if done:
                # etat terminal : pas de bootstrap sur le futur
                target = r
                a_next = None
            else:
                if method == "sarsa":
                    a_next = epsilon_greedy(x_next, W, epsilon, n_actions, rng)
                    target = r + gamma * q_hat(x_next, W)[a_next]
                elif method == "qlearning":
                    a_next = epsilon_greedy(x_next, W, epsilon, n_actions, rng)  # action reellement jouee ensuite
                    target = r + gamma * np.max(q_hat(x_next, W))  # mais le bootstrap utilise le max
                else:
                    raise ValueError(method)

            delta = target - q_hat(x, W)[a]
            W[a] += alpha * delta * x

            x, a = x_next, a_next
            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(eps_end, epsilon * eps_decay)

    env.close()
    return np.array(episode_rewards), W


def moving_average(values, window=20):
    return np.convolve(values, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    N_EPISODES = 800

    print("Entrainement SARSA...")
    rewards_sarsa, W_sarsa = train("sarsa", n_episodes=N_EPISODES, seed=0)
    print(f"  Reward moyen (100 derniers episodes) : {rewards_sarsa[-100:].mean():.1f}")

    print("Entrainement Q-learning...")
    rewards_qlearning, W_qlearning = train("qlearning", n_episodes=N_EPISODES, seed=0)
    print(f"  Reward moyen (100 derniers episodes) : {rewards_qlearning[-100:].mean():.1f}")

    # -----------------------------------------------------------------
    # Plot comparatif
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(moving_average(rewards_sarsa), label="SARSA (moyenne glissante 20 ep.)")
    plt.plot(moving_average(rewards_qlearning), label="Q-learning (moyenne glissante 20 ep.)")
    plt.axhline(500, color="gray", linestyle="--", linewidth=1, label="Max (CartPole-v1 = 500)")
    plt.xlabel("Episode")
    plt.ylabel("Reward total (= nb de steps survecus)")
    plt.title("CartPole-v1 -- SARSA vs Q-learning (VFA lineaire)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cartpole_sarsa_vs_qlearning.png", dpi=120)
    print("Graphe sauvegarde : cartpole_sarsa_vs_qlearning.png")
