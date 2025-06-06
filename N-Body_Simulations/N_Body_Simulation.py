import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from matplotlib.animation import FuncAnimation, FFMpegWriter

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

def main():
    N = 200  # Number of particles
    t = 0  # Initial time
    tEnd = 10.0 * 365.25 * 24 * 3600  # Total simulation time in seconds (10 years)
    dt = 1 * 24 * 3600  # Time step in seconds (1 day)
    softening = 1e9  # Softening length in meters
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    box_size = 1e11  # Box size limit in meters (100 billion meters)

    #np.random.seed(42)
    mass = 5.972e24 * np.ones(N)  # Mass of each particle (Earth mass in kg), shape (N,)
    R = 6.378e7  # Radius of a particle (Earth-like)
    min_dist = 2 * R  # Minimum allowed distance between particles
    pos = initialize_positions(N, box_size, min_dist)  # Confined to a cubic region
    vel = np.random.randn(N, 3) * 300  # Random velocities in m/s (close to Earth's orbital speed)
    
    vel -= np.mean(mass[:, None] * vel, axis=0) / np.mean(mass)  # Center of mass correction
    acc_a = acc(pos, mass, G, softening)
    Nt = int(np.ceil(tEnd / dt))
    
    # Setup 3D figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the writer
    writer = FFMpegWriter(fps=30)
    
    # Create a function for updating the plot
    def update(i):
        nonlocal pos, vel, acc_a, t
        vel += acc_a * dt / 2.0
        pos += vel * dt
        acc_a = acc(pos, mass, G, softening)
        vel += acc_a * dt / 2.0
        t += dt
        
        ax.clear()
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, color='blue')
        ax.set_xlim([-box_size, box_size])
        ax.set_ylim([-box_size, box_size])
        ax.set_zlim([-box_size, box_size])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"N-Body Simulation (t = {t/3.154e7:.2f} years)")
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=Nt, repeat=False)

    # Save the animation as an MP4 file
    ani.save('N_Body_simulation_nonzero_velocity.mp4', writer=writer)
    
    plt.show()

if __name__ == "__main__":
    main()
