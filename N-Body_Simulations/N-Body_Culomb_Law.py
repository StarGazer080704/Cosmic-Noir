import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
from math import sqrt

def acc(q, m, softening, charges=None, ke=8.9875e9):
    n = m.shape[0]
    accel = np.zeros((n, 3))

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        for j in range(n):
            if j == i:
                continue
            dx, dy, dz = q[j, 0] - q[i, 0], q[j, 1] - q[i, 1], q[j, 2] - q[i, 2]
            dist_sq = dx**2 + dy**2 + dz**2 + softening**2
            dist = sqrt(dist_sq)
            inv_dist_cube = 1.0 / (dist_sq * dist)

            f_e = 0.0
            if charges is not None:
                f_e = -ke * charges[i] * charges[j] / (dist_sq * dist)

            total_force = f_e / m[i]

            ax += total_force * dx
            ay += total_force * dy
            az += total_force * dz

        accel[i] = [ax, ay, az]
    return accel

def main():
    N = 50
    t = 0
    dt = 1e-15
    tEnd = 1e-12  # Shorter for a faster test render
    softening = 1e-10
    box_size = 1e-8

    m_p = 1.673e-27
    q_e = -1.602e-19
    q_p = 1.602e-19

    mass = m_p * np.ones(N)
    pos = np.random.uniform(-box_size, box_size, (N, 3))
    vel = np.random.randn(N, 3) * 1e3

    charges = np.random.uniform(q_e, q_p, N)
    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)

    acc_a = acc(pos, mass, softening, charges)
    Nt = int(np.ceil(tEnd / dt))

    colors = ['red' if q > 0 else 'blue' for q in charges]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set up FFMpegWriter
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    with writer.saving(fig, "nbody_simulation_CL.mp4", dpi=100):

        for i in range(Nt):
            vel += acc_a * dt / 2.0
            pos += vel * dt
            acc_a = acc(pos, mass, softening, charges)
            vel += acc_a * dt / 2.0
            t += dt

            if i % 10 == 0:
                ax.clear()
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, color=colors)
                ax.set_xlim([-box_size, box_size])
                ax.set_ylim([-box_size, box_size])
                ax.set_zlim([-box_size, box_size])
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f"N-Body Simulation (Culomb's Law) (t = {t:.2e} s)")

                writer.grab_frame()

if __name__ == "__main__":
    main()
