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
        if np.all(np.linalg.norm(candidate - p) >= min_dist for p in positions):
            positions.append(candidate)

    return np.array(positions)

def compute_energy(q, v, m, G, softening):
    N = len(m)
    KE = 0.5 * np.sum(m * np.sum(v**2, axis=1))
    
    PE = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = q[j] - q[i]
            dist = np.sqrt(np.sum(dx**2) + softening**2)
            PE -= G * m[i] * m[j] / dist

    return KE, PE, KE + PE


def main():
    N = 100  # Number of particles
    t = 0  # Initial time
    tEnd = 10000.0 * 365.25 * 24 * 3600  # Total simulation time in seconds
    dt = 300 * 24 * 3600  # Time step in seconds (15 days)
    softening = 1e10  # Softening length in meters
    G = 6.67430e-11  # Gravitational constant
    box_size = 2e15  # Box size in meters
    mass = 2e30 * np.random.uniform(0.1, 10, N)  # Randomized mass
    R = 7e8  # Radius of a particle (Sun-like)
    r = R * (mass / 2e30) ** 0.8  # Radius of each particle based on mass
    
    energies_kin = []
    energies_pot = []
    energies_tot = []
    times = []

    #np.random.seed(42)
    pos = initialize_positions(N, box_size, r)
    vel = np.random.randn(N, 3) * 3000*2  # Initial random velocity

    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)  # Center of mass velocity correction
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(Nt):
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt
        if i % 10 == 0:
            KE, PE, TE = compute_energy(pos, vel, mass, G, softening)
            energies_kin.append(KE)
            energies_pot.append(PE)
            energies_tot.append(TE)
            times.append(t / 3.154e7)  # Convert time to years for plotting        

        if i % 10 == 0:
            ax.clear()
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, color='orange')
            ax.set_xlim([-box_size, box_size])
            ax.set_ylim([-box_size, box_size])
            ax.set_zlim([-box_size, box_size])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f"100-Suns Simulation (t = {t/3.154e7:.2f} years)")
            plt.pause(0.001)
    plt.figure(figsize=(8, 5))
    plt.plot(times, energies_kin, label='Kinetic Energy')
    plt.plot(times, energies_pot, label='Potential Energy')
    plt.plot(times, energies_tot, label='Total Energy', linestyle='--')
    plt.xlabel('Time (years)')
    plt.ylabel('Energy (Joules)')
    plt.title('Energy Conservation in N-body Simulation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    plt.show()

if __name__ == "__main__":
    main()

