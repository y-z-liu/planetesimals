import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

# ---------------- Tunable parameters ----------------
N_INIT         = 1000          # initial number of bodies
T_END_YEARS    = 100           # total evolution time in years

R_FACTOR       = 5             # collision‐radius scaling factor
TOT_MASS_RATIO = 100           # total mass in units of Earth mass

RHO_AU         = 0.02          # initial orbital scatter ratio
RHO_MASS       = 0.2           # initial mass scatter ratio

DT_MIN         = 1.0e2         # minimum timestep (s)
DT_MAX         = 1.0e5         # maximum timestep (s)

COLOR          = "Blues"       # colormap for scatter
PLOT_SCALE     = 1e-6          # scale factor for marker sizes
# ----------------------------------------------------

# ---------------- Physical constants ----------------
G        = 6.67430e-11       # gravitational constant [m^3 kg^-1 s^-2]
M_STAR   = 1.989e30          # mass of central star (kg)
M_EARTH  = 5.972e24          # Earth mass (kg)
AU       = 1.496e11          # astronomical unit (m)
R_STAR   = 6.957e8           # radius of central star (m)
R_EARTH  = 6.371e6           # radius of Earth (m)
# ----------------------------------------------------


@njit(parallel=True)
def compute_accelerations(pos, masses):
    """
    Compute pairwise accelerations plus central‐star gravity.
    """
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        xi, yi = pos[i]
        axi = 0.0
        ayi = 0.0

        # mutual gravity
        for j in range(n):
            if i == j:
                continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            d3 = (dx*dx + dy*dy)**1.5
            axi += -G * masses[j] * dx / d3
            ayi += -G * masses[j] * dy / d3

        # gravity from the star at the origin
        r = np.hypot(xi, yi)
        axi += -G * M_STAR * xi / (r**3)
        ayi += -G * M_STAR * yi / (r**3)

        acc[i,0] = axi
        acc[i,1] = ayi

    return acc


@njit
def leapfrog_step(pos, vel, masses, dt):
    """
    Advance positions & velocities by one step (velocity‐Verlet).
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
    Spatial hashing: find candidate collision pairs.
    """
    n = pos.shape[0]
    gx = np.empty(n, np.int64)
    gy = np.empty(n, np.int64)
    for i in prange(n):
        gx[i] = int(pos[i,0] // cell_size)
        gy[i] = int(pos[i,1] // cell_size)

    # count & collect
    cnt = 0
    for i in prange(n):
        for j in range(i+1, n):
            if abs(gx[i]-gx[j])<=1 and abs(gy[i]-gy[j])<=1:
                cnt += 1

    pairs = np.empty((cnt,2), np.int64)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(gx[i]-gx[j])<=1 and abs(gy[i]-gy[j])<=1:
                pairs[idx,0] = i
                pairs[idx,1] = j
                idx += 1

    return pairs


@njit(parallel=True)
def min_time_to_collision(pos, vel, radii, pairs, R_factor):
    """
    Solve for collision times among candidate pairs.
    """
    m = pairs.shape[0]
    times = np.empty(m, dtype=np.float64)
    for k in prange(m):
        i, j = pairs[k]
        dx = pos[i,0] - pos[j,0]
        dy = pos[i,1] - pos[j,1]
        rvx = vel[i,0] - vel[j,0]
        rvy = vel[i,1] - vel[j,1]
        Rsum = (radii[i] + radii[j]) * R_factor

        a = rvx*rvx + rvy*rvy
        b = 2.0*(dx*rvx + dy*rvy)
        c = dx*dx + dy*dy - Rsum*Rsum

        if a == 0.0:
            times[k] = 0.0 if c <= 0.0 else np.inf
        else:
            disc = b*b - 4*a*c
            if disc < 0.0:
                times[k] = np.inf
            else:
                sd = np.sqrt(disc)
                t1 = (-b - sd)/(2*a)
                t2 = (-b + sd)/(2*a)
                tcol = np.inf
                if t1>=0.0: tcol = t1
                if 0.0<=t2<tcol: tcol = t2
                times[k] = tcol

    return times


@njit(parallel=True)
def detect_sun_absorption(pos, vel, dt, Rstar_scaled):
    """
    Flag bodies that hit the star within dt.
    """
    n = pos.shape[0]
    out = np.zeros(n, np.bool_)
    for i in prange(n):
        x, y = pos[i]
        vx, vy = vel[i]
        v2 = vx*vx + vy*vy
        if v2 > 0.0:
            t_hit = - (x*vx + y*vy) / v2
            t_hit = 0.0 if t_hit<0 else dt if t_hit>dt else t_hit
        else:
            t_hit = 0.0
        xx = x + vx*t_hit
        yy = y + vy*t_hit
        if xx*xx + yy*yy <= Rstar_scaled*Rstar_scaled:
            out[i] = True
    return out


@njit
def resolve_collisions_and_absorption(pos, vel, masses, radii,
                                      pairs, tcols, dt,
                                      R_factor, Rstar_scaled):
    """
    Merge colliding bodies, remove those absorbed by the star.
    """
    n = masses.shape[0]
    parent = np.arange(n, dtype=np.int64)

    # union-find
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # union any pair that collides within dt
    for k in range(pairs.shape[0]):
        if tcols[k] <= dt:
            i, j = pairs[k]
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

    # re-label groups
    labels = np.empty(n, np.int64)
    for i in range(n):
        labels[i] = find(i)

    # map roots → new indices
    idx_map = -np.ones(n, np.int64)
    new_n = 0
    for i in range(n):
        r = labels[i]
        if idx_map[r] == -1:
            idx_map[r] = new_n
            new_n += 1

    # accumulate into merged arrays
    new_pos = np.zeros((new_n,2), dtype=np.float64)
    new_vel = np.zeros((new_n,2), dtype=np.float64)
    new_mass = np.zeros(new_n, dtype=np.float64)
    new_rad  = np.zeros(new_n, dtype=np.float64)

    for i in range(n):
        g = idx_map[labels[i]]
        m = masses[i]
        new_mass[g] += m
        new_pos[g]  += pos[i] * m
        new_vel[g]  += vel[i] * m

    # finalize COM positions & radii
    for g in range(new_n):
        mt = new_mass[g]
        new_pos[g] /= mt
        new_vel[g] /= mt
        new_rad[g] = (mt / M_EARTH)**(1/3) * R_EARTH

    # remove any that fall into the star
    absorbed = detect_sun_absorption(new_pos, new_vel, dt, Rstar_scaled)
    survivors = (~absorbed).sum()

    final_pos = np.zeros((survivors,2), dtype=np.float64)
    final_vel = np.zeros((survivors,2), dtype=np.float64)
    final_mass= np.zeros(survivors, dtype=np.float64)
    final_rad = np.zeros(survivors, dtype=np.float64)

    idx = 0
    for i in range(new_n):
        if not absorbed[i]:
            final_pos[idx]  = new_pos[i]
            final_vel[idx]  = new_vel[i]
            final_mass[idx] = new_mass[i]
            final_rad[idx]  = new_rad[i]
            idx += 1

    return final_pos, final_vel, final_mass, final_rad


def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    """
    Distribute bodies in a ring with random orbital and mass scatter.
    Positions: ring at 1 AU ± RHO_AU * AU
    Velocities: circular ± speed_variation
    Masses: mean ± RHO_MASS fraction
    """
    mass_mean = total_mass_ratio * M_EARTH / n

    # radial scatter
    r0     = AU + np.random.uniform(-RHO_AU*AU, RHO_AU*AU, n)
    theta  = np.random.uniform(0, 2*np.pi, n)
    pos    = np.vstack([r0*np.cos(theta), r0*np.sin(theta)]).T

    # circular speed + scatter
    v_circ = np.sqrt(G * M_STAR / r0)
    v_mag  = v_circ * (1 + np.random.uniform(-speed_variation,
                                             speed_variation, n))
    vel    = np.vstack([-v_mag*np.sin(theta), v_mag*np.cos(theta)]).T

    # correct mass scatter (fractional)
    masses = mass_mean * (1 + np.random.uniform(-RHO_MASS,
                                                 RHO_MASS, n))
    radii  = (masses / M_EARTH)**(1/3) * R_EARTH

    return pos, vel, masses, radii


def simulate(n=N_INIT, total_mass_ratio=TOT_MASS_RATIO,
             t_end_years=T_END_YEARS):
    """
    Yields (time, positions, masses) step-by-step until t_end_years
    or only one body remains.
    """
    pos, vel, masses, radii = init_bodies
    t = 0.0
    t_end = t_end_years * 365 * 86400.0
    dt = DT_MIN

    while t < t_end and len(masses) > 1:
        # neighbor search box size
        max_v = np.max(np.linalg.norm(vel, axis=1))
        cell = 2 * np.max(radii) * R_FACTOR + max_v * dt
        pairs = neighbour_pairs(pos, cell)

        # collision times
        if pairs.size:
            tcols = min_time_to_collision(pos, vel, radii, pairs, R_FACTOR)
        else:
            tcols = np.empty(0, dtype=np.float64)

        # adapt dt
        future = tcols[tcols > dt]
        dt_next = np.clip((future.min()-dt)*2, DT_MIN, DT_MAX) if future.size else DT_MAX

        # integrate
        pos, vel = leapfrog_step(pos, vel, masses, dt)

        # merge & remove
        sun_scaled = R_STAR * R_FACTOR
        pos, vel, masses, radii = resolve_collisions_and_absorption(
            pos, vel, masses, radii, pairs, tcols, dt,
            R_FACTOR, sun_scaled
        )

        t += dt
        yield t, pos.copy(), masses.copy()
        dt = dt_next


def animate(sim_gen):
    """
    Build a side-by-side animation:
      • Left:  scatter of bodies around the star (log-colored by mass)
      • Right: histogram of mass distribution (log-y, log-x) with dynamic y-axis limits
    """
    # pull first frame
    t0, pos0, m0 = next(sim_gen)
    em0 = m0 / M_EARTH

    # figure & axes
    fig, (ax_sc, ax_h) = plt.subplots(1, 2, figsize=(12,6))

    # --- left: scatter + star ---
    ax_sc.add_patch(plt.Circle((0,0), R_STAR, color='red'))
    ax_sc.set_xlim(-2*AU, 2*AU)
    ax_sc.set_ylim(-2*AU, 2*AU)
    ax_sc.set_aspect('equal')
    ax_sc.xaxis.set_major_locator(plt.MultipleLocator(AU))
    ax_sc.yaxis.set_major_locator(plt.MultipleLocator(AU))
    ax_sc.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x/AU:.0f} AU"))
    ax_sc.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y/AU:.0f} AU"))

    mean_m = TOT_MASS_RATIO * M_EARTH / N_INIT
    norm  = LogNorm(vmin=mean_m*(1-RHO_MASS), vmax=mean_m*(1+RHO_MASS)*N_INIT)
    r0 = (em0)**(1/3)*R_EARTH
    s0 = (r0 * PLOT_SCALE)**2

    scat = ax_sc.scatter(pos0[:,0], pos0[:,1],
                         c=m0, cmap=COLOR, norm=norm, s=s0)

    # --- right: histogram ---
    min_em = (1-RHO_MASS)*TOT_MASS_RATIO / N_INIT
    max_em = (1+RHO_MASS)*TOT_MASS_RATIO
    bins   = np.logspace(np.log10(min_em), np.log10(max_em), 30)

    cnt0, _ = np.histogram(em0, bins=bins)
    cnt0 = np.where(cnt0>0, cnt0, 1)  # ensure non-zero

    bars = ax_h.bar(bins[:-1], cnt0,
                    width=np.diff(bins), align='edge',
                    edgecolor='black')

    ax_h.set_xscale('log')
    ax_h.set_yscale('log')
    ax_h.set_xlabel("Mass (Earth masses)")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Mass Distribution")
    
    # Set initial y-axis limits based on first frame
    ax_h.set_ylim(1, cnt0.max() * 1.2)

    def update(frame):
        t, pos, m = frame
        em = m / M_EARTH

        # update scatter
        r = (em)**(1/3)*R_EARTH
        s = (r * PLOT_SCALE)**2
        scat.set_offsets(pos)
        scat.set_array(m)
        scat.set_sizes(s)
        ax_sc.set_title(f"t = {t/86400/365:.2f} yr   N = {len(m)}")

        # update histogram
        cnt, _ = np.histogram(em, bins=bins)
        cnt = np.where(cnt>0, cnt, 1e-8)
        for rect, h in zip(bars, cnt):
            rect.set_height(h)

        # dynamically adjust y-axis limit
        ax_h.set_ylim(1e-1, cnt.max() * 1.2)

        return (scat, *bars)

    return FuncAnimation(fig, update,
                         frames=sim_gen,
                         interval=20, blit=False,
                         cache_frame_data=False)

if __name__ == "__main__":
    sim = simulate()
    anim = animate(sim)
    plt.show()