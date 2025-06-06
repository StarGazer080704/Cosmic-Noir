import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def rk45_step(pos, vel, mass, G, softening, dt):
    def deriv(y):
        q = y[:N*3].reshape(N, 3)
        v = y[N*3:].reshape(N, 3)
        a = acc(q, mass, G, softening)
        return np.concatenate([v.flatten(), a.flatten()])

    y = np.concatenate([pos.flatten(), vel.flatten()])
    k1 = dt * deriv(y)
    k2 = dt * deriv(y + 0.25 * k1)
    k3 = dt * deriv(y + (3/32)*k1 + (9/32)*k2)
    k4 = dt * deriv(y + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = dt * deriv(y + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = dt * deriv(y - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)

    y_next = y + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6

    pos_next = y_next[:N*3].reshape(N, 3)
    vel_next = y_next[N*3:].reshape(N, 3)
    return pos_next, vel_next

def main():
    global N  # Make N accessible to rk45_step
    N = 3
    t = 0
    tEnd = 1000.0 * 365.25 * 24 * 3600
    dt = 5 * 24 * 3600
    softening = 1e2
    G = 6.67430e-11
    box_size = 1e11

    #np.random.seed(42)
    mass = 5.972e24 * np.ones(N)
    R = 6.378e7
    min_dist = 2 * R
    pos = initialize_positions(N, box_size, min_dist)
    vel = np.random.randn(N, 3) * 30*0
    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)

    Nt = int(np.ceil(tEnd / dt))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(Nt):
        pos, vel = rk45_step(pos, vel, mass, G, softening, dt)
        t += dt

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

    plt.show()

if __name__ == "__main__":
    main()