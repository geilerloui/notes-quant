"""
FrozenLake-v1 : Policy Iteration "a la lettre" du fichier 01 (Programmation Dynamique).

On a acces au modele exact (P(s'|s,a), R(s,a)) via env.unwrapped.P, donc pas besoin
de MC/TD ici -- c'est le terrain naturel de la DP.

Boucle :
    A. Policy Evaluation  : iterer Bellman a pi fixee jusqu'a convergence de V.
    B. Policy Improvement : Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s'), puis pi = greedy(Q).
    Repeter jusqu'a ce que pi ne bouge plus.

On garde des snapshots (V, Q, pi) a 3 moments : premiere iteration, iteration
intermediaire, iteration finale -- pour visualiser l'evolution sans noyer
l'affichage sous 100 plots.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

GAMMA = 0.99
THETA = 1e-8          # seuil de convergence de Policy Evaluation
MAX_OUTER_ITER = 50    # nb max de tours Policy Iteration
ACTION_ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}  # LEFT, DOWN, RIGHT, UP (convention Gymnasium)


# ---------------------------------------------------------------------------
# Recuperation du modele exact
# ---------------------------------------------------------------------------
def get_model(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = env.unwrapped.P  # P[s][a] = liste de (proba, s_suivant, reward, done)
    return P, n_states, n_actions


# ---------------------------------------------------------------------------
# A. Policy Evaluation (pseudo-code fichier 01, section II.A)
# ---------------------------------------------------------------------------
def policy_evaluation(pi, P, n_states, n_actions, gamma=GAMMA, theta=THETA, max_iter=10000):
    V = np.zeros(n_states)
    for _ in range(max_iter):
        delta = 0.0
        new_V = np.zeros(n_states)
        for s in range(n_states):
            v = 0.0
            for a in range(n_actions):
                prob_a = pi[s, a]
                if prob_a == 0.0:
                    continue
                for prob, s_next, r, done in P[s][a]:
                    v += prob_a * prob * (r + gamma * V[s_next] * (not done))
            new_V[s] = v
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    return V


# ---------------------------------------------------------------------------
# Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s')
# ---------------------------------------------------------------------------
def compute_q_from_v(V, P, n_states, n_actions, gamma=GAMMA):
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            q = 0.0
            for prob, s_next, r, done in P[s][a]:
                q += prob * (r + gamma * V[s_next] * (not done))
            Q[s, a] = q
    return Q


# ---------------------------------------------------------------------------
# B. Policy Improvement : politique deterministe = argmax_a Q(s,a)
# ---------------------------------------------------------------------------
def policy_improvement(Q):
    n_states, n_actions = Q.shape
    new_pi = np.zeros((n_states, n_actions))
    best_actions = np.argmax(Q, axis=1)
    new_pi[np.arange(n_states), best_actions] = 1.0
    return new_pi


# ---------------------------------------------------------------------------
# C. Policy Iteration complet, avec sauvegarde de snapshots
# ---------------------------------------------------------------------------
def policy_iteration(env, gamma=GAMMA, max_outer_iter=MAX_OUTER_ITER):
    P, n_states, n_actions = get_model(env)
    pi = np.ones((n_states, n_actions)) / n_actions  # politique initiale uniforme

    history = []  # liste de dicts {iter, V, Q, pi}
    converged_at = None

    for k in range(1, max_outer_iter + 1):
        V = policy_evaluation(pi, P, n_states, n_actions, gamma)
        Q = compute_q_from_v(V, P, n_states, n_actions, gamma)
        new_pi = policy_improvement(Q)

        history.append({"iter": k, "V": V.copy(), "Q": Q.copy(), "pi": new_pi.copy()})

        if np.array_equal(new_pi, pi):
            converged_at = k
            break
        pi = new_pi

    return history, converged_at


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_V_grid(ax, V, grid_size, title, holes=(), goal=None):
    grid = V.reshape(grid_size, grid_size)
    im = ax.imshow(grid, cmap="cool", vmin=0, vmax=max(V.max(), 0.01))
    for s in range(len(V)):
        r, c = divmod(s, grid_size)
        color = "black"
        face = None
        if s in holes:
            face = "black"
        elif s == goal:
            face = "magenta"
        if face:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=face))
        ax.text(c, r, f"{V[s]:.2f}", ha="center", va="center",
                color="white" if s in holes else "black", fontsize=9)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_policy_arrows(ax, pi, grid_size, title, holes=(), goal=None):
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    for s in range(pi.shape[0]):
        r, c = divmod(s, grid_size)
        face = "black" if s in holes else ("magenta" if s == goal else "lightcyan")
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=face, ec="gray"))
        if s not in holes and s != goal:
            a = np.argmax(pi[s])
            ax.text(c, r, ACTION_ARROWS[a], ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_Q_heatmap(ax, Q, title):
    im = ax.imshow(Q, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("action (0=L,1=D,2=R,3=U)")
    ax.set_ylabel("etat s")
    ax.set_xticks(range(Q.shape[1]))
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            ax.text(a, s, f"{Q[s, a]:.2f}", ha="center", va="center", color="white", fontsize=6)


# ---------------------------------------------------------------------------
# Le modele (P, R) : FIXE, connu depuis le debut, ne change jamais pendant
# les iterations -- contrairement a V, Q, pi qui evoluent. On le montre a part.
# ---------------------------------------------------------------------------
def build_R_matrix(P, n_states, n_actions):
    """R(s,a) = recompense esperee -- meme forme que Q(s,a), (n_states, n_actions)."""
    R = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            R[s, a] = sum(prob * r for prob, s_next, r, done in P[s][a])
    return R


def build_P_tensor(P, n_states, n_actions):
    """P_tensor[a, s, s'] = P(s' | s, a) -- une matrice de transition S x S par action."""
    P_tensor = np.zeros((n_actions, n_states, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for prob, s_next, r, done in P[s][a]:
                P_tensor[a, s, s_next] += prob
    return P_tensor


def plot_R_heatmap(ax, R, title):
    im = ax.imshow(R, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("action (0=L,1=D,2=R,3=U)")
    ax.set_ylabel("etat s")
    ax.set_xticks(range(R.shape[1]))
    for s in range(R.shape[0]):
        for a in range(R.shape[1]):
            ax.text(a, s, f"{R[s, a]:.2f}", ha="center", va="center", color="white", fontsize=6)


# Sur FrozenLake slippery, P(s'|s,a) ne prend que 4 valeurs possibles : 0, 1/3, 2/3, 1
# (0, 1, 2 ou 3 "tranches" de 1/3 qui menent au meme etat suivant). Une colorbar continue
# n'a donc aucun sens -- on discretise en 4 couleurs + une legende.
PROB_LEVELS = [0.0, 1 / 3, 2 / 3, 1.0]
PROB_COLORS = ["#1a1a2e", "#4361ee", "#f77f00", "#d62828"]  # noir, bleu, orange, rouge -- 4 teintes bien distinctes


def _prob_cmap_norm():
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(PROB_COLORS)
    boundaries = [-1 / 6, 1 / 6, 3 / 6, 5 / 6, 7 / 6]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def plot_P_heatmap(ax, P_sa, title):
    """
    P_sa : matrice (n_states, n_states) = P(s' | s, a) pour une action a fixee.
    Affichee transposee : s' en ligne (axe y), s en colonne (axe x, "en bas").
    """
    n_states = P_sa.shape[0]
    cmap, norm = _prob_cmap_norm()
    im = ax.imshow(P_sa.T, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("s (etat courant)")
    ax.set_ylabel("s' (etat suivant)")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.tick_params(axis="both", labelsize=7)
    return im


def make_model_figure(env, n_states, n_actions, filename):
    """Figure statique : R(s,a) + les 4 matrices P(.|.,a), une par action."""
    import matplotlib.patches as mpatches

    P = env.unwrapped.P
    R = build_R_matrix(P, n_states, n_actions)
    P_tensor = build_P_tensor(P, n_states, n_actions)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))

    plot_R_heatmap(axes[0], R, "R(s,a)")

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    for a in range(n_actions):
        plot_P_heatmap(axes[a + 1], P_tensor[a], f"P(s'|s, a={action_names[a]})")

    # Legende discrete (4 couleurs = 4 valeurs de probabilite possibles) a la place d'une colorbar continue
    legend_patches = [
        mpatches.Patch(color=PROB_COLORS[i], label=f"{PROB_LEVELS[i]:.2f}")
        for i in range(len(PROB_LEVELS))
    ]
    fig.legend(handles=legend_patches, title="probabilite", loc="center right",
               bbox_to_anchor=(1.02, 0.5), fontsize=9, title_fontsize=10)

    fig.suptitle("Le modele (P, R) -- FIXE, connu depuis le debut, ne change jamais pendant Policy Iteration",
                 fontsize=13, y=1.03)
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Sauvegarde : {filename}")


def make_combined_figure(history, holes, goal, grid_size, filename):
    """
    Une seule figure : chaque ligne = une iteration (debut / milieu / fin),
    chaque colonne = V(s) | Q(s,a) | politique pi(a|s).
    Permet de lire d'un coup d'oeil comment les trois objets evoluent ensemble.
    """
    n = len(history)
    idxs = sorted(set([0, n // 2, n - 1]))

    fig, axes = plt.subplots(
        len(idxs), 3, figsize=(15, 5 * len(idxs)),
        gridspec_kw={"width_ratios": [1, 1.3, 1]},
    )
    if len(idxs) == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["V(s)", "Q(s,a)", "Politique pi(a|s)"]

    for row, idx in enumerate(idxs):
        snap = history[idx]
        it = snap["iter"]
        plot_V_grid(axes[row, 0], snap["V"], grid_size, f"V(s) -- iter {it}", holes, goal)
        plot_Q_heatmap(axes[row, 1], snap["Q"], f"Q(s,a) -- iter {it}")
        plot_policy_arrows(axes[row, 2], snap["pi"], grid_size, f"pi(a|s) -- iter {it}", holes, goal)

    for ax, col_title in zip(axes[0], col_titles):
        ax.annotate(col_title, xy=(0.5, 1.15), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")

    fig.suptitle("Policy Iteration sur FrozenLake -- evolution de V, pi, Q", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Sauvegarde : {filename}")


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    grid_size = int(np.sqrt(env.observation_space.n))

    # Carte par defaut 4x4 : "SFFF/FHFH/FFFH/HFFG" -> trous en 5,7,11,12 ; but en 15.
    desc = env.unwrapped.desc.astype(str)
    holes = [r * grid_size + c for r in range(grid_size) for c in range(grid_size) if desc[r][c] == "H"]
    goal = [r * grid_size + c for r in range(grid_size) for c in range(grid_size) if desc[r][c] == "G"][0]

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    make_model_figure(env, n_states, n_actions, "frozenlake_model_PR.png")

    history, converged_at = policy_iteration(env)
    print(f"Convergence apres {converged_at} iterations (sur {len(history)} enregistrees)")

    make_combined_figure(history, holes, goal, grid_size, "frozenlake_combined_snapshots.png")
