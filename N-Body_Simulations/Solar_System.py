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
def main():
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    softening = 1e9  # Larger softening to stabilize solar-scale simulation
    dt = 24 * 3600  # 1 day in seconds
    tEnd = 365.25 * 24 * 3600 * 10  # Simulate for 10 years
    t = 0

    Bodies = [
        {"name": "Sun", "distance": 0, "velocity": 0, "mass": 1.989e30},
        {"name": "Mercury", "distance": 5.791e10, "velocity": 4.787e4, "mass": 3.3011e23},
        {"name": "Venus",   "distance": 1.082e11, "velocity": 3.502e4, "mass": 4.8675e24},
        {"name": "Earth",   "distance": 1.496e11, "velocity": 2.978e4, "mass": 5.97237e24},
        {"name": "Mars",    "distance": 2.279e11, "velocity": 2.407e4, "mass": 6.4171e23},
        {"name": "Jupiter", "distance": 7.785e11, "velocity": 1.307e4, "mass": 100*1.8982e27},
        {"name": "Saturn",  "distance": 1.43e12,  "velocity": 0.969e4, "mass": 5.6834e26},
        {"name": "Uranus",  "distance": 2.87e12,  "velocity": 0.681e4, "mass": 8.6810e25},
        {"name": "Neptune", "distance": 4.50e12,  "velocity": 0.543e4, "mass": 1.02413e26}
    ]
    colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'goldenrod', 'lightblue', 'darkblue']
    s_s = [20, 2, 3, 3, 3, 10, 9, 8, 8]

    N = len(Bodies)
    mass = np.array([b["mass"] for b in Bodies])

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    for i,b in enumerate(Bodies):
        if i == 0:
            pos[i] = np.array([0.0, 0.0, 0.0])
            vel[i] = np.array([0.0, 0.0, 0.0])
        else:
            pos[i] = np.array([b["distance"], 0.0, 0.0])
            vel[i] = np.array([0.0, b["velocity"], 0.0])
    
    vel -= np.sum((mass[:, None]*vel), axis=0)/np.sum(mass)
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd/dt))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    
    for i in range(Nt):
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt
        
        
        if i % 10 == 0:
            ax.clear()
            for j in range(N):
                ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], color=colors[j], s=s_s[j])
                ax.text(pos[j, 0], pos[j, 1], pos[j, 2], Bodies[j]["name"], color=colors[j], fontsize=7, horizontalalignment='center')
            ax.set_xlim([-2e12, 2e12])
            ax.set_ylim([-2e11, 2e11])
            ax.set_zlim([-1e12, 1e12])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f"Solar System Simulation (t = {t/3.154e7:.2f} years)")
            plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()