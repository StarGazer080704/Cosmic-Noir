import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def acc(q, m, G, softening):
    n = m.shape[0]
    accel = np.zeros((n, 2))  # 2D acceleration
    
    for i in range(n):
        ax, ay = 0.0, 0.0
        for j in range(n):
            if j == i:
                continue
            dx, dy = q[j, 0] - q[i, 0], q[j, 1] - q[i, 1]
            dist = sqrt(dx**2 + dy**2 + softening**2)
            force = G * m[j] / (dist**3)
            ax += force * dx
            ay += force * dy
        accel[i] = [ax, ay]
    return accel

def initialize_positions(N, box_size, min_dist):
    positions = []
    while len(positions) < N:
        candidate = np.random.uniform(-box_size, box_size, 2)
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
    N = 200  # Number of particles
    t = 0
    tEnd = 7.0 * 365.25 * 24 * 3600
    dt = 8 * 24 * 3600
    softening = 1e2
    G = 6.67430e-11
    box_size = 1e11

    mass = 5.972e24 * np.ones(N)
    R = 6.378e7
    min_dist = 2 * R
    pos = initialize_positions(N, box_size, min_dist)
    vel = np.random.randn(N, 2) * 0

    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))

    # Setup 2D figure
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(Nt):
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt

        if i % 10 == 0:
            ax.clear()
            ax.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
            ax.set_xlim([-box_size, box_size])
            ax.set_ylim([-box_size, box_size])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f"2D N-Body Simulation (t = {t/3.154e7:.2f} years)")
            plt.pause(0.001)

    # -------- FoF Halo Detection --------
    linking_length = 0.2 * (2 * box_size) / N
    clusters = friends_of_friends(pos, linking_length)
    filtered_clusters = [c for c in clusters if len(c) >= 5]
    print(f"Found {len(filtered_clusters)} halos with ≥5 stars.")

    # -------- Final Visualization --------
    ax.clear()
    colors = plt.cm.jet(np.linspace(0, 1, len(filtered_clusters)))
    for cluster, color in zip(filtered_clusters, colors):
        cluster_positions = pos[cluster]
        ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], s=10, color=color)

    ax.set_xlim([-box_size, box_size])
    ax.set_ylim([-box_size, box_size])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Identified 2D Halos (Total = {len(filtered_clusters)})")
    plt.show()

if __name__ == "__main__":
    main()
