import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from numba import njit, prange

# ---------------- Tunable parameters ----------------
R_FACTOR       = 10             # collision radius scaling factor
N_INIT         = 1000          # initial number of bodies
TOT_MASS_RATIO = 100           # total mass divided by Earth mass
T_END_YEARS    = 100           # total evolution time in years

# adaptive time‐step bounds (seconds)
DT_MIN         = 1.0e4
DT_MAX         = 1.0e5

# solar wind at 1 AU dynamic pressure [Pa]
SW_PRESSURE    = 2.0e-9

# assumed body density [kg/m^3] (typical rocky body)
BODY_DENSITY   = 3000.0
# ----------------------------------------------------

# ---------------- Physical constants ----------------
G        = 6.67430e-11       # gravitational constant [m^3 kg^-1 s^-2]
M_STAR   = 1.989e30          # mass of central star (kg)
M_EARTH  = 5.972e24          # Earth mass (kg)
AU       = 1.496e11          # astronomical unit (m)
DELTA_AU = 0.01 * AU         # initial orbital scatter (m)
R_STAR   = 6.957e8           # radius of central star (m)
# ----------------------------------------------------

def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    """
    Initialize positions, velocities, masses, and radii of bodies.
    Bodies start in roughly circular orbits with small scatter.
    """
    mass_per_body = total_mass_ratio * M_EARTH / n
    r0 = AU + np.random.uniform(-DELTA_AU, DELTA_AU, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    pos = np.vstack((r0 * np.cos(theta), r0 * np.sin(theta))).T

    # circular speed plus optional variation
    v_circ = np.sqrt(G * M_STAR / r0)
    v_mag = v_circ * (1 + np.random.uniform(-speed_variation,
                                             speed_variation, n))
    vel = np.vstack((-v_mag * np.sin(theta),
                     v_mag * np.cos(theta))).T

    masses = np.full(n, mass_per_body)
    radii = (mass_per_body / M_EARTH) ** (1/3) * 6.371e6  # Earth radius scale
    return pos, vel, masses, np.full(n, radii)

@njit(parallel=True)
def compute_accelerations(pos, masses, radii):
    """
    Compute accelerations from mutual gravity, star gravity, and solar wind.
    """
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        xi, yi = pos[i, 0], pos[i, 1]
        ax = ay = 0.0

        # mutual gravity
        for j in range(n):
            if i == j: continue
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            d3 = (dx*dx + dy*dy)**1.5
            ax += -G * masses[j] * dx / d3
            ay += -G * masses[j] * dy / d3

        # gravity from central star
        r = np.hypot(xi, yi)
        ax += -G * M_STAR * xi / (r**3)
        ay += -G * M_STAR * yi / (r**3)

        # solar wind acceleration
        coeff = 0.75 * (SW_PRESSURE / (BODY_DENSITY * radii[i])) * AU**2
        ax += coeff * xi / (r**3)
        ay += coeff * yi / (r**3)

        acc[i, 0] = ax
        acc[i, 1] = ay
    return acc

@njit
def leapfrog_step(pos, vel, masses, radii, dt):
    """
    Advance positions and velocities by one time step (velocity Verlet).
    """
    a0 = compute_accelerations(pos, masses, radii)
    vel_half = vel + 0.5 * dt * a0
    pos_new  = pos + dt * vel_half
    a1 = compute_accelerations(pos_new, masses, radii)
    vel_new = vel_half + 0.5 * dt * a1
    return pos_new, vel_new

@njit(parallel=True)
def neighbour_pairs(pos, cell_size):
    """
    Find candidate pairs whose grid cells are within one cell of each other.
    """
    n = pos.shape[0]
    grid_x = np.empty(n, np.int64)
    grid_y = np.empty(n, np.int64)
    for i in prange(n):
        grid_x[i] = int(pos[i, 0] // cell_size)
        grid_y[i] = int(pos[i, 1] // cell_size)

    # count pairs
    cnt = 0
    for i in prange(n):
        xi, yi = grid_x[i], grid_y[i]
        for j in range(i+1, n):
            dx = xi - grid_x[j]
            dy = yi - grid_y[j]
            if abs(dx) <= 1 and abs(dy) <= 1:
                cnt += 1

    pairs = np.empty((cnt, 2), dtype=np.int64)
    idx = 0
    for i in range(n):
        xi, yi = grid_x[i], grid_y[i]
        for j in range(i+1, n):
            dx = xi - grid_x[j]
            dy = yi - grid_y[j]
            if abs(dx) <= 1 and abs(dy) <= 1:
                pairs[idx, 0] = i
                pairs[idx, 1] = j
                idx += 1
    return pairs

@njit(parallel=True)
def min_time_to_collision(pos, vel, radii, pairs, R_factor):
    """
    For each candidate pair, solve |r0 + v_rel * t| = R_sum
    and return smallest non-negative root, or inf if none.
    """
    n_pairs = pairs.shape[0]
    times = np.empty(n_pairs, dtype=np.float64)
    for k in prange(n_pairs):
        i, j = pairs[k, 0], pairs[k, 1]
        r0x = pos[i, 0] - pos[j, 0]
        r0y = pos[i, 1] - pos[j, 1]
        vrelx = vel[i, 0] - vel[j, 0]
        vrely = vel[i, 1] - vel[j, 1]
        R_sum = (radii[i] + radii[j]) * R_factor

        a = vrelx*vrelx + vrely*vrely
        b = 2.0 * (r0x*vrelx + r0y*vrely)
        c = r0x*r0x + r0y*r0y - R_sum*R_sum

        if a == 0.0:
            times[k] = 0.0 if c <= 0.0 else np.inf
        else:
            disc = b*b - 4*a*c
            if disc < 0.0:
                times[k] = np.inf
            else:
                sd = np.sqrt(disc)
                t1 = (-b - sd) / (2*a)
                t2 = (-b + sd) / (2*a)
                tcol = np.inf
                if t1 >= 0.0: tcol = t1
                if t2 >= 0.0 and t2 < tcol: tcol = t2
                times[k] = tcol
    return times

@njit(parallel=True)
def detect_sun_absorption(pos, vel, dt, R_star_scaled):
    """
    Detect bodies absorbed by the star within one timestep.
    """
    n = pos.shape[0]
    absorb = np.zeros(n, np.bool_)
    for i in prange(n):
        r0x, r0y = pos[i,0], pos[i,1]
        vrelx, vrely = vel[i,0], vel[i,1]
        v2 = vrelx*vrelx + vrely*vrely
        if v2 > 0.0:
            dot = r0x*vrelx + r0y*vrely
            tstar = -dot / v2
            tstar = 0.0 if tstar < 0 else (dt if tstar > dt else tstar)
        else:
            tstar = 0.0
        dx = r0x + vrelx * tstar
        dy = r0y + vrely * tstar
        if dx*dx + dy*dy <= R_star_scaled*R_star_scaled:
            absorb[i] = True
    return absorb

def handle_collisions(pos, vel, masses, radii, dt):
    """
    Merge colliding bodies and remove star‐absorbed ones.
    Collisions are those pairs with collision_time <= dt.
    """
    n = len(masses)
    if n <= 1:
        return pos, vel, masses, radii

    # build neighbor list with current dt
    max_speed = np.max(np.linalg.norm(vel, axis=1))
    cell_size = 2 * np.max(radii) * R_FACTOR + max_speed * dt
    pairs = neighbour_pairs(pos, cell_size)

    # find which pairs actually collide within dt
    if pairs.size > 0:
        tcols = min_time_to_collision(pos, vel, radii, pairs, R_FACTOR)
        collide_mask = tcols <= dt
        coll_pairs = pairs[collide_mask]
    else:
        coll_pairs = np.empty((0,2), dtype=np.int64)

    # union‐find to group collisions
    parent = np.arange(n, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in coll_pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # merge groups
    new_pos = []; new_vel = []; new_m = []; new_r = []
    for idxs in groups.values():
        if len(idxs) == 1:
            i = idxs[0]
            new_pos.append(pos[i]); new_vel.append(vel[i])
            new_m.append(masses[i]); new_r.append(radii[i])
        else:
            ids   = np.array(idxs, dtype=np.int64)
            m_tot = masses[ids].sum()
            p_cm  = (pos[ids] * masses[ids,None]).sum(axis=0) / m_tot
            v_cm  = (vel[ids] * masses[ids,None]).sum(axis=0) / m_tot
            r_cm  = (m_tot / M_EARTH)**(1/3) * 6.371e6
            new_pos.append(p_cm); new_vel.append(v_cm)
            new_m.append(m_tot); new_r.append(r_cm)

    new_pos = np.array(new_pos)
    new_vel = np.array(new_vel)
    new_m   = np.array(new_m)
    new_r   = np.array(new_r)

    # remove bodies absorbed by the star
    sun_rad_scaled = R_STAR * R_FACTOR
    absorbed = detect_sun_absorption(new_pos, new_vel, dt, sun_rad_scaled)
    if np.any(absorbed):
        keep = ~absorbed
        new_pos = new_pos[keep]
        new_vel = new_vel[keep]
        new_m   = new_m[keep]
        new_r   = new_r[keep]

    return new_pos, new_vel, new_m, new_r

def simulate(n=N_INIT, total_mass_ratio=TOT_MASS_RATIO,
             t_end_years=T_END_YEARS):
    """
    Generator yielding (t, pos, masses, vmin, vmax) each step.
    dt is chosen from the time‐to‐collision estimate.
    """
    pos, vel, masses, radii = initialize_bodies(n, total_mass_ratio)
    t = 0.0
    t_end = t_end_years * 365 * 86400.0

    while t < t_end and len(masses) > 1:
        # neighbor list for collision timing
        max_speed = np.max(np.linalg.norm(vel, axis=1))
        cell_size = 2 * np.max(radii) * R_FACTOR + max_speed * DT_MAX
        pairs = neighbour_pairs(pos, cell_size)

        # compute min time to any collision
        if pairs.size > 0:
            tcols = min_time_to_collision(pos, vel, radii, pairs, R_FACTOR)
            t_coll_min = np.min(tcols)
        else:
            t_coll_min = np.inf

        # choose dt from collision time
        dt = np.clip(t_coll_min, DT_MIN, DT_MAX)

        # advance integrator
        pos, vel = leapfrog_step(pos, vel, masses, radii, dt)

        # merge collisions & remove star hits
        pos, vel, masses, radii = handle_collisions(
            pos, vel, masses, radii, dt)

        # emit for animation
        vmin = masses.min() * 0.1
        vmax = masses.max() * 10.0
        t += dt
        yield t, pos.copy(), masses.copy(), vmin, vmax

def animate(sim_gen):
    """
    Visualize the simulation using matplotlib animation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(0, 0, marker=(12,1,0), markersize=15, color='orange')
    ax.set_xlim(-2*AU, 2*AU)
    ax.set_ylim(-2*AU, 2*AU)
    ax.set_aspect('equal')

    scat = ax.scatter([], [], s=12)

    def update(frame):
        t, pos, masses, vmin, vmax = frame
        scat.set_offsets(pos)
        scat.set_array(masses)
        scat.set_cmap('Greys')
        scat.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f"t = {t/86400/365:.2f} yr   N = {len(masses)}")
        return scat

    return FuncAnimation(fig, update, frames=sim_gen,
                         interval=20, blit=False,
                         cache_frame_data=False)

if __name__ == "__main__":
    sim_gen = simulate()
    anim    = animate(sim_gen)
    plt.show()