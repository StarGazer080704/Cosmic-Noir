import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

def acc(q, m, G, softening):
    n = m.shape[0]
    accel = np.zeros((n, 3))  # Correct shape (N, 3)
    
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
    N = 200  # Number of particles
    t = 0  # Initial time
    tEnd = 7.0 * 365.25 * 24 * 3600  # Total simulation time in seconds (10 years)
    dt = 8 * 24 * 3600  # Time step in seconds (5 days)
    softening = 1e2  # Softening length in meters
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    box_size = 1e11  # Box size limit in meters (100 billion meters)

    #np.random.seed(42)
    mass = 5.972e24 * np.ones(N)  # Mass of each particle (Earth mass in kg)
    R = 6.378e7  # Radius of a particle (Earth-like)
    min_dist = 2 * R  # Minimum allowed distance between particles
    pos = initialize_positions(N, box_size, min_dist)
    vel = np.random.randn(N, 3) * 30*0  # Random velocities in m/s

    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)  # Center of mass correction
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))

    # Setup 3D figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(Nt):
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt

        # Real-time plotting (optional)
        if i % 10 == 0:
            ax.clear()
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, color='blue')
            ax.set_xlim([-box_size, box_size])
            ax.set_ylim([-box_size, box_size])
            ax.set_zlim([-box_size, box_size])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f"N-Body Simulation (t = {t/3.154e7:.2f} years)")
            plt.pause(0.001)

    # -------- Friends-of-Friends: Halo Detection --------
    linking_length = 0.2*(box_size*2)/N  
    clusters = friends_of_friends(pos, linking_length)
    #clusters = [c for c in clusters if len(c) >= 2]  # Filter small halos

    print(f"Found {len(clusters)} halos with ≥5 stars.")

    # -------- Final Visualization of Halos --------
    ax.clear()
    colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))
    for cluster, color in zip(clusters, colors):
        cluster_positions = pos[cluster]
        ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], cluster_positions[:, 2], s=10, color=color)

    ax.set_xlim([-box_size, box_size])
    ax.set_ylim([-box_size, box_size])
    ax.set_zlim([-box_size, box_size])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"Identified Halos (Total = {len(clusters)})")
    plt.show()

if __name__ == "__main__":
    main()
