import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from math import sqrt

def acc(q, m, G, softening):
    n = m.shape[0]
    accel = np.zeros((n, 3))
    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        for j in range(n):
            if j == i:
                continue
            dx, dy, dz = q[j, 0] - q[i, 0], q[j, 1] - q[i, 1], q[j, 2] - q[i, 2]
            dist = sqrt(dx**2 + dy**2 + dz**2 + softening**2)
            force = G * m[j] / (dist**3)
            ax += force * dx
            ay += force * dy
            az += force * dz
        accel[i] = [ax, ay, az]
    return accel

def initialize_positions(N, box_size, min_dist):
    positions = []
    while len(positions) < N:
        candidate = np.random.uniform(-box_size, box_size, 3)
        if all(np.linalg.norm(candidate - p) >= min_dist for p in positions):
            positions.append(candidate)
    return np.array(positions)

def friends_of_friends(positions, linking_length):
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            current = queue.pop()
            distances = np.linalg.norm(positions - positions[current], axis=1)
            neighbors = np.where((distances < linking_length) & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                cluster.append(neighbor)
                queue.append(neighbor)
        clusters.append(cluster)
    return clusters

def main():
    N = 200
    t = 0
    tEnd = 7.0 * 365.25 * 24 * 3600
    dt = 1 * 24 * 3600
    softening = 1e2
    G = 6.67430e-11
    box_size = 1e11

    mass = 5.972e24 * np.ones(N)
    R = 6.378e7
    min_dist = 2 * R
    pos = initialize_positions(N, box_size, min_dist)
    vel = np.random.randn(N, 3) * 0  # Optional randomness
    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))

    linking_length = 2e10

    cluster_history = []
    pos_history = []
    time_points = []
    halo_counts = []

    for i in range(Nt):
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt

        if i % 10 == 0:
            clusters = friends_of_friends(pos, linking_length)
            clusters = [c for c in clusters if len(c) >= 2]
            cluster_history.append(clusters)
            pos_history.append(np.copy(pos))
            time_points.append(t / 3.154e7)  # in years
            halo_counts.append(len(clusters))
            print(f"Time: {time_points[-1]:.2f} yrs, Halos: {len(clusters)}")

    # ---------- Plot Halo Count Over Time ----------
    plt.figure(figsize=(8, 4))
    plt.plot(time_points, halo_counts, marker='o')
    plt.xlabel('Time (years)')
    plt.ylabel('Number of Halos')
    plt.title('Evolution of Halo Count Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Animate Halo Evolution ----------
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        pos = pos_history[frame]
        clusters = cluster_history[frame]
        colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))
        for cluster, color in zip(clusters, colors):
            cluster_positions = pos[cluster]
            ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], cluster_positions[:, 2], s=10, color=color)
        ax.set_xlim([-box_size, box_size])
        ax.set_ylim([-box_size, box_size])
        ax.set_zlim([-box_size, box_size])
        ax.set_title(f"t = {time_points[frame]:.2f} years")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

    ani = animation.FuncAnimation(fig, update, frames=len(pos_history), interval=300)

    # Save as MP4
    ani.save("halo_evolution.mp4", writer='ffmpeg', fps=5, dpi=200)

    plt.show()

if __name__ == "__main__":
    main()
