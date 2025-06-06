import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from matplotlib.animation import FFMpegWriter

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
        if np.all(np.linalg.norm(candidate - p) >= min_dist for p in positions):
            positions.append(candidate)

    return np.array(positions)


def main():
    N = 100
    t = 0
    tEnd = 10000.0 * 365.25 * 24 * 3600
    dt = 300 * 24 * 3600
    softening = 1e10
    G = 6.67430e-11
    box_size = 2e15
    mass = 2e30 * np.random.uniform(0.1, 10, N)
    R = 7e8
    r = R * (mass / 2e30) ** 0.8

    #np.random.seed(42)
    pos = initialize_positions(N, box_size, r)
    vel = np.random.randn(N, 3) * 3000*2
    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    writer = FFMpegWriter(fps=30)

    with writer.saving(fig, "100_Sun_Simulation_2.mp4", dpi=100):
        for i in range(Nt):
            vel += acc_a * dt / 2.0
            pos += vel * dt
            acc_a = acc(pos, mass, G, softening)
            vel += acc_a * dt / 2.0
            t += dt

            if i % 10 == 0:
                ax.clear()
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, color='orange')
                ax.set_xlim([-box_size, box_size])
                ax.set_ylim([-box_size, box_size])
                ax.set_zlim([-box_size, box_size])
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f"100-Sun Clusters (t = {t/3.154e7:.2f} years)")
                writer.grab_frame()

    plt.show()

if __name__ == "__main__":
    main()
