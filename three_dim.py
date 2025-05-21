import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from numba import njit, prange

# ---------------- Tunable parameters ----------------
N_INIT         = 1000          # initial number of bodies
T_END_YEARS    = 100           # total evolution time in years

R_FACTOR       = 5             # collision radius scaling factor
TOT_MASS_RATIO = 10           # total mass divided by Earth mass

RHO_AU         = 0.2          # initial variation ratio of orbit
RHO_MASS       = 0.5           # initial variation ratio of mass
X_RATIO        = 1          # X-axis shrinking ratio
Y_RATIO        = 1           # Y-axis shrinking ratio

# adaptive time‐step bounds (seconds)
DT_MIN         = 1.0e2
DT_MAX         = 1.0e5

# visualization
COlOR          = "Blues"
INIT_SIZE      = 1
SCOPE_RATIO    = 2
# ----------------------------------------------------

# ---------------- Physical constants ----------------
G        = 6.67430e-11       # gravitational constant [m^3 kg^-1 s^-2]
M_STAR   = 1.989e30          # mass of central star (kg)
M_EARTH  = 5.972e24          # Earth mass (kg)
AU       = 1.496e11          # astronomical unit (m)
R_STAR   = 6.957e8           # radius of central star (m)
R_EARTH  = 6.371e6           # radius of earth (m)
# ----------------------------------------------------

@njit(parallel=True)
def compute_accelerations(pos, masses):
    """
    Compute accelerations from mutual gravity and central star gravity.
    """
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        xi, yi = pos[i, 0], pos[i, 1]
        ax = 0.0
        ay = 0.0
        # mutual gravity
        for j in range(n):
            if i == j:
                continue
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            d3 = (dx*dx + dy*dy)**1.5
            ax += -G * masses[j] * dx / d3
            ay += -G * masses[j] * dy / d3
        # gravity from central star
        r = (xi*xi + yi*yi)**0.5
        ax += -G * M_STAR * xi / (r**3)
        ay += -G * M_STAR * yi / (r**3)
        acc[i, 0] = ax
        acc[i, 1] = ay
    return acc

@njit
def leapfrog_step(pos, vel, masses, dt):
    """
    Advance positions and velocities by one time step (velocity Verlet).
    """
    a0 = compute_accelerations(pos, masses)
    vel_half = vel + 0.5 * dt * a0
    pos_new  = pos + dt * vel_half
    a1 = compute_accelerations(pos_new, masses)
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

    # count qualifying pairs
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
    For each candidate pair, solve |r0 + v_rel * t| = R_sum and return times.
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

@njit
def resolve_collisions_and_absorption(pos, vel, masses, radii,
                                      pairs, tcols, dt,
                                      R_factor, R_star_scaled):
    """
    Merge colliding bodies, remove star-absorbed ones, return updated arrays.
    """
    n = masses.shape[0]
    # union-find parent array
    parent = np.arange(n, dtype=np.int64)

    def find_parent(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # union colliding pairs
    for k in range(pairs.shape[0]):
        if tcols[k] <= dt:
            i, j = pairs[k,0], pairs[k,1]
            pi = find_parent(i)
            pj = find_parent(j)
            if pi != pj:
                parent[pj] = pi

    # find root for each body
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        labels[i] = find_parent(i)

    # map each root to a new group index
    new_index = np.full(n, -1, dtype=np.int64)
    new_n = 0
    for i in range(n):
        root = labels[i]
        if new_index[root] == -1:
            new_index[root] = new_n
            new_n += 1

    # allocate arrays for merged bodies
    new_pos = np.zeros((new_n,2), dtype=np.float64)
    new_vel = np.zeros((new_n,2), dtype=np.float64)
    new_mass = np.zeros(new_n, dtype=np.float64)
    new_rad  = np.zeros(new_n, dtype=np.float64)

    # accumulate mass and momentum
    for i in range(n):
        gi = new_index[labels[i]]
        m = masses[i]
        new_mass[gi] += m
        new_pos[gi,0] += pos[i,0] * m
        new_pos[gi,1] += pos[i,1] * m
        new_vel[gi,0] += vel[i,0] * m
        new_vel[gi,1] += vel[i,1] * m

    # finalize center of mass and compute radius
    for g in range(new_n):
        m_tot = new_mass[g]
        new_pos[g,0] /= m_tot
        new_pos[g,1] /= m_tot
        new_vel[g,0] /= m_tot
        new_vel[g,1] /= m_tot
        new_rad[g] = (m_tot / M_EARTH)**(1/3) * R_EARTH

    # detect and remove star-absorbed bodies
    absorb = detect_sun_absorption(new_pos, new_vel, dt, R_star_scaled)
    # count survivors
    cnt = 0
    for i in range(new_n):
        if not absorb[i]:
            cnt += 1

    # allocate final arrays
    final_pos = np.zeros((cnt,2), dtype=np.float64)
    final_vel = np.zeros((cnt,2), dtype=np.float64)
    final_mass= np.zeros(cnt, dtype=np.float64)
    final_rad = np.zeros(cnt, dtype=np.float64)

    idx = 0
    for i in range(new_n):
        if not absorb[i]:
            final_pos[idx,0]  = new_pos[i,0]
            final_pos[idx,1]  = new_pos[i,1]
            final_vel[idx,0]  = new_vel[i,0]
            final_vel[idx,1]  = new_vel[i,1]
            final_mass[idx]   = new_mass[i]
            final_rad[idx]    = new_rad[i]
            idx += 1

    return final_pos, final_vel, final_mass, final_rad

def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    """
    Initialize positions, velocities, masses, and radii of bodies.
    """
    mass_mean = total_mass_ratio * M_EARTH / n
    r0 = AU + np.random.uniform(-RHO_AU * AU, RHO_AU * AU, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    pos = np.vstack((X_RATIO * r0 * np.cos(theta), Y_RATIO * r0 * np.sin(theta))).T

    v_circ = np.sqrt(G * M_STAR / r0)
    v_mag = v_circ * (1 + np.random.uniform(-speed_variation,
                                             speed_variation, n))
    vel = np.vstack((-v_mag * np.sin(theta),
                     v_mag * np.cos(theta))).T

    masses = mass_mean + np.random.uniform(-RHO_MASS, RHO_MASS, n)
    radii  = (masses / M_EARTH)**(1/3) * R_EARTH

    return pos, vel, masses, radii

def simulate(init_bodies, n=N_INIT, total_mass_ratio=TOT_MASS_RATIO,
             t_end_years=T_END_YEARS):
    """
    Generator yielding (t, pos, masses) each step.
    Uses previous dt for grid, collision detection, and integration,
    and computes next dt during collision detection.
    """
    pos, vel, masses, radii = init_bodies
    t = 0.0
    t_end = t_end_years * 365 * 86400.0
    dt = DT_MIN  # start with minimum timestep

    while t < t_end and len(masses) > 1:
        # build neighbor list using current dt
        max_speed = np.max(np.linalg.norm(vel, axis=1))
        cell_size = 2 * np.max(radii) * R_FACTOR + max_speed * dt
        pairs = neighbour_pairs(pos, cell_size)

        # compute collision times for all candidate pairs
        if pairs.size > 0:
            tcols = min_time_to_collision(pos, vel, radii, pairs, R_FACTOR)
        else:
            tcols = np.empty(0, dtype=np.float64)

        # compute next dt based on non-colliding pairs
        future = tcols[tcols > dt]
        if future.size > 0:
            dt_next = np.clip((future.min() - dt) * 2.0, DT_MIN, DT_MAX)
        else:
            dt_next = DT_MAX

        # advance positions and velocities
        pos, vel = leapfrog_step(pos, vel, masses, dt)

        # merge collisions and remove star-absorbed bodies
        sun_rad_scaled = R_STAR * R_FACTOR
        pos, vel, masses, radii = resolve_collisions_and_absorption(
            pos, vel, masses, radii, pairs, tcols, dt,
            R_FACTOR, sun_rad_scaled)

        t += dt
        print(f"Time: {t/86400/365:.2f} yr | Bodies: {len(masses)} | dt: {dt:.1e} s")
        yield t, pos.copy(), masses.copy()

        # update dt for next iteration
        dt = dt_next


def animate(sim_gen):
    """
    Visualize the simulation using matplotlib animation.
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(0, 0, marker=(12,1,0), markersize=15, color='red')
    ax.set_xlim(-SCOPE_RATIO*AU, SCOPE_RATIO*AU)
    ax.set_ylim(-SCOPE_RATIO*AU, SCOPE_RATIO*AU)
    ax.set_aspect('equal')

    mass_mean = TOT_MASS_RATIO * M_EARTH / N_INIT# 这是初始状态平均质量
    norm = LogNorm(vmin=mass_mean * 0.1,
                   vmax=mass_mean * 0.5 * N_INIT)# 这里是采用了log归一化来实现颜色映射
    s_init = INIT_SIZE
    scat = ax.scatter([], [], s = s_init,
                      c=[], cmap= COlOR, norm=norm)

    def update(frame):
        t, pos, masses = frame
        s_current = np.power(masses / mass_mean, 2.0/3.0) * INIT_SIZE
        scat.set_offsets(pos)
        scat.set_array(masses)
        scat.set_sizes(s_current)
        ax.set_title(f"t = {t/86400/365:.2f} yr   N = {len(masses)}")
        return scat

    return FuncAnimation(fig, update, frames=sim_gen,
                         interval=20, blit=False,
                         cache_frame_data=False)

if __name__ == "__main__":
    init_bodies = initialize_bodies(N_INIT, TOT_MASS_RATIO)
    sim_gen = simulate(init_bodies)
    anim    = animate(sim_gen)
    plt.show()