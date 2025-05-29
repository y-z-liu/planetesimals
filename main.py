import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

# ---------------- Tunable parameters ----------------
N_INIT         = 1000              # initial number of micro-planets
T_END_YEARS    = 10000             # total integration time [years]
DRAW_SKIP      = [1e3, 3e5, 3e2]   # {normal interval, except every _, slow down _}

R_FACTOR       = 5                 # collision-radius scaling factor
TOT_MASS_RATIO = 10                # total planetary mass, in Earth masses

RHO_AU         = 0.02              # orbital scatter fraction
RHO_MASS       = 0.2               # mass scatter fraction

DT_MIN         = 1.0e2             # minimum timestep [s]
DT_MAX         = 1.0e5             # maximum timestep [s]

COLOR          = "Blues"           # colormap for planet scatter
PLOT_SCALE     = 2e-6              # planet scatter size
PLT_STYLE      = "Solarize_Light2" # style for matplotlib

SEED           = None              # random seed for reproducibility (None to disable)
SAVE_YEARS     = list(np.arange(1000)/10) + list(range(100,10001,100))
                                   # years at which to capture snapshots for GIF;
                                   # set to [] to disable GIF output and enable animate
GIF_FILENAME   = 'selected_frames.gif'
# ----------------------------------------------------

# ---------------- Physical constants ----------------
G        = 6.67430e-11       # gravitational constant [m^3 kg^-1 s^-2]
M_STAR   = 1.989e30          # mass of the star [kg]
M_EARTH  = 5.972e24          # Earth mass [kg]
AU       = 1.496e11          # astronomical unit [m]
R_STAR   = 6.957e8           # stellar radius [m]
R_EARTH  = 6.371e6           # Earth radius [m]
# ----------------------------------------------------


@njit(parallel=True)
def compute_accelerations(pos, masses):
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        xi, yi = pos[i,0], pos[i,1]
        axi, ayi = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            inv_r3 = 1.0 / (dx*dx + dy*dy)**1.5
            axi += -G * masses[j] * dx * inv_r3
            ayi += -G * masses[j] * dy * inv_r3
        acc[i,0], acc[i,1] = axi, ayi
    return acc


@njit
def leapfrog_step(pos, vel, masses, dt):
    a0 = compute_accelerations(pos, masses)
    vel_half = vel + 0.5 * dt * a0
    pos_new = pos + dt * vel_half
    a1 = compute_accelerations(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * a1
    return pos_new, vel_new


@njit(parallel=True)
def neighbour_pairs(pos, cell_size):
    n = pos.shape[0]
    gx = np.empty(n, np.int64)
    gy = np.empty(n, np.int64)
    for i in prange(n):
        gx[i] = int(pos[i,0] // cell_size)
        gy[i] = int(pos[i,1] // cell_size)

    cnt = 0
    for i in prange(n):
        for j in range(i+1, n):
            if abs(gx[i] - gx[j]) <= 1 and abs(gy[i] - gy[j]) <= 1:
                cnt += 1

    pairs = np.empty((cnt, 2), np.int64)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(gx[i] - gx[j]) <= 1 and abs(gy[i] - gy[j]) <= 1:
                pairs[idx,0] = i
                pairs[idx,1] = j
                idx += 1

    return pairs


@njit(parallel=True)
def min_time_to_collision(pos, vel, radii, pairs, R_factor):
    m = pairs.shape[0]
    times = np.empty(m, dtype=np.float64)
    for k in prange(m):
        i, j = pairs[k]
        dx = pos[i,0] - pos[j,0]
        dy = pos[i,1] - pos[j,1]
        dvx = vel[i,0] - vel[j,0]
        dvy = vel[i,1] - vel[j,1]
        Rsum = (radii[i] + radii[j]) * R_factor

        a = dvx*dvx + dvy*dvy
        b = 2.0 * (dx*dvx + dy*dvy)
        c = dx*dx + dy*dy - Rsum*Rsum

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
                if t1 >= 0.0:
                    tcol = t1
                if 0.0 <= t2 < tcol:
                    tcol = t2
                times[k] = tcol
    return times


@njit
def resolve_collisions(pos, vel, masses, radii,
                       pairs, tcols, dt, R_factor):
    n = masses.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Union pairs that collide within dt
    for k in range(pairs.shape[0]):
        i, j = pairs[k]
        if tcols[k] <= dt:
            pi, pj = find(i), find(j)
            if pi != pj:
                if pi == 0 or pj == 0:
                    parent[max(pi, pj)] = 0
                else:
                    parent[pj] = pi

    # Flatten groups
    labels = np.empty(n, np.int64)
    for i in range(n):
        labels[i] = find(i)

    idx_map = -np.ones(n, np.int64)
    new_n = 0
    for i in range(n):
        root = labels[i]
        if idx_map[root] == -1:
            idx_map[root] = new_n
            new_n += 1

    new_pos  = np.zeros((new_n, 2), dtype=np.float64)
    new_vel  = np.zeros((new_n, 2), dtype=np.float64)
    new_mass = np.zeros(new_n, dtype=np.float64)
    new_rad  = np.zeros(new_n, dtype=np.float64)

    for i in range(n):
        g = idx_map[labels[i]]
        m = masses[i]
        new_mass[g] += m
        new_pos[g]  += pos[i] * m
        new_vel[g]  += vel[i] * m

    star_group = idx_map[find(0)]
    for g in range(new_n):
        mt = new_mass[g]
        new_pos[g] /= mt
        new_vel[g] /= mt
        if g == star_group:
            new_rad[g] = R_STAR
        else:
            new_rad[g] = (mt / M_EARTH)**(1/3) * R_EARTH

    return new_pos, new_vel, new_mass, new_rad


def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    mass_mean = total_mass_ratio * M_EARTH / n
    r0     = AU + np.random.uniform(-RHO_AU*AU, RHO_AU*AU, n)
    theta  = np.random.uniform(0, 2*np.pi, n)
    pos_p  = np.vstack([r0*np.cos(theta), r0*np.sin(theta)]).T
    v_circ = np.sqrt(G * M_STAR / r0)
    speed  = v_circ * (1 + np.random.uniform(-speed_variation, speed_variation, n))
    vel_p  = np.vstack([-speed*np.sin(theta), speed*np.cos(theta)]).T
    masses_p = mass_mean * (1 + np.random.uniform(-RHO_MASS, RHO_MASS, n))
    radii_p  = (masses_p / M_EARTH)**(1/3) * R_EARTH

    pos    = np.vstack((np.zeros((1,2)), pos_p))
    vel    = np.vstack((np.zeros((1,2)), vel_p))
    masses = np.concatenate((np.array([M_STAR]), masses_p))
    radii  = np.concatenate((np.array([R_STAR]), radii_p))

    return pos, vel, masses, radii


@njit(parallel=True)
def compute_total_energy(pos, vel, masses):
    n = pos.shape[0]
    KE = 0.0
    for i in range(n):
        KE += 0.5 * masses[i] * (vel[i,0]**2 + vel[i,1]**2)

    PE = 0.0
    for i in prange(n):
        mi = masses[i]
        xi, yi = pos[i,0], pos[i,1]
        for j in range(i+1, n):
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            PE += -G * mi * masses[j] / np.hypot(dx, dy)

    return KE + PE


def simulate(n=N_INIT, total_mass_ratio=TOT_MASS_RATIO,
             t_end_years=T_END_YEARS, seed=SEED):
    """
    Generator yielding (t, positions, masses, velocities, energy).
    Optionally seeds numpy RNG for reproducible runs.
    """
    if seed is not None:
        np.random.seed(seed)

    pos, vel, masses, radii = initialize_bodies(n, total_mass_ratio)
    t = 0.0
    t_end = t_end_years * 365 * 86400.0
    dt = DT_MIN

    while t < t_end and masses.size > 1:
        max_v = np.max(np.linalg.norm(vel, axis=1))
        cell = 2 * np.max(radii) * R_FACTOR + 2 * max_v * dt

        pairs = neighbour_pairs(pos, cell)
        tcols = min_time_to_collision(pos, vel, radii, pairs, R_FACTOR) if pairs.size else np.empty(0)

        future = tcols[tcols > dt]
        dt_next = (np.clip((future.min() - dt)*2, DT_MIN, DT_MAX)
                   if future.size else DT_MAX)

        pos, vel = leapfrog_step(pos, vel, masses, dt)
        pos, vel, masses, radii = resolve_collisions(
            pos, vel, masses, radii, pairs, tcols, dt, R_FACTOR
        )

        t += dt
        E = compute_total_energy(pos, vel, masses)
        yield t, pos.copy(), masses.copy(), vel.copy(), E

        dt = dt_next


def animate(sim_gen, draw_interval=DRAW_SKIP,
            save_years=SAVE_YEARS, gif_filename=GIF_FILENAME):
    """
    If save_years is non-empty, captures snapshots at those years and
    writes a GIF. Otherwise, shows the standard interactive animation.
    """
    # Helper: set up figure, axes, initial artists, and update() function.
    def setup_animation(first_frame):
        t0, pos0, m0, vel0, E0 = first_frame
        star_x0, star_y0 = pos0[0]
        pos_p0, m_p0 = pos0[1:], m0[1:]

        plt.style.use(PLT_STYLE)
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2,
                              width_ratios=[2,1],
                              height_ratios=[3,3,3],
                              wspace=0.3, hspace=0.7)

        # Scatter panel
        ax_sc = fig.add_subplot(gs[:,0])
        star_marker = ax_sc.scatter([star_x0], [star_y0],
                                    s=300, marker=(12,1,0), c='red')
        ax_sc.set_xlim(-2.5*AU, 2.5*AU)
        ax_sc.set_ylim(-2.5*AU, 2.5*AU)
        ax_sc.set_aspect('equal')
        ax_sc.xaxis.set_major_locator(plt.MultipleLocator(AU))
        ax_sc.yaxis.set_major_locator(plt.MultipleLocator(AU))
        ax_sc.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x/AU:.0f} AU'))
        ax_sc.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y/AU:.0f} AU'))

        mean_mass = TOT_MASS_RATIO * M_EARTH / N_INIT
        norm = LogNorm(vmin=mean_mass*0.1, vmax=mean_mass*N_INIT)
        r0 = (m_p0 / M_EARTH)**(1/3) * R_EARTH
        s0 = (r0 * PLOT_SCALE)**2
        scat = ax_sc.scatter(pos_p0[:,0], pos_p0[:,1],
                             c=m_p0, cmap=COLOR, norm=norm, s=s0)

        # Histogram panel
        ax_h = fig.add_subplot(gs[0,1])
        min_em = (1-RHO_MASS) * TOT_MASS_RATIO / N_INIT
        max_em = (1+RHO_MASS) * TOT_MASS_RATIO
        bins = np.logspace(np.log10(min_em), np.log10(max_em), 30)
        cnt0, _ = np.histogram(m_p0 / M_EARTH, bins=bins)
        cnt0 = np.where(cnt0>0, cnt0, 1)
        bars = ax_h.bar(bins[:-1], cnt0,
                        width=np.diff(bins), align='edge', edgecolor='black')
        ax_h.set_xscale('log')
        ax_h.set_yscale('log')
        ax_h.set_xlabel('M (Earth masses)')
        ax_h.set_ylabel('Count')
        ax_h.set_title('Mass Distribution')
        ax_h.set_ylim(1e-1, cnt0.max()*1.2)

        # Count vs time
        ax_n = fig.add_subplot(gs[1,1])
        times = [t0 / (86400*365)]
        counts = [m_p0.size]
        line_count, = ax_n.plot(times, counts, 'b-')
        ax_n.set_xlabel('t (years)')
        ax_n.set_ylabel('N (bodies)')
        ax_n.set_title('Number of Planetesimals')

        # Energy vs time
        ax_e = fig.add_subplot(gs[2,1])
        energies = [E0]
        line_energy, = ax_e.plot(times, energies, 'm-')
        ax_e.set_xlabel('t (years)')
        ax_e.set_ylabel('Energy (J)')
        ax_e.set_title('Total Mechanical Energy')

        def update(frame):
            t, pos, m, vel, E = frame
            star_marker.set_offsets(pos[0])
            pos_p, m_p = pos[1:], m[1:]

            # scatter
            r = (m_p / M_EARTH)**(1/3) * R_EARTH
            s = (r * PLOT_SCALE)**2
            scat.set_offsets(pos_p)
            scat.set_array(m_p)
            scat.set_sizes(s)
            ax_sc.set_title(f't = {t/(86400*365):.2f} yr   N = {m_p.size}')

            # histogram
            cnt, _ = np.histogram(m_p / M_EARTH, bins=bins)
            cnt = np.where(cnt>0, cnt, 1e-8)
            for rect, h in zip(bars, cnt):
                rect.set_height(h)
            ax_h.set_ylim(1e-1, cnt.max()*1.2)

            # count curve
            times.append(t/(86400*365))
            counts.append(m_p.size)
            line_count.set_data(times, counts)
            ax_n.set_xlim(0, times[-1]*1.05)
            ax_n.set_ylim(0, max(counts)*1.05)

            # energy curve
            energies.append(E)
            line_energy.set_data(times, energies)
            ax_e.set_xlim(0, times[-1]*1.05)
            ax_e.set_ylim(min(energies), max(energies))

            return (star_marker, scat, *bars, line_count, line_energy)

        return fig, update

    # Branch: snapshots-to-GIF vs interactive
    if save_years:
        # Run sim and collect frames at target years
        year_sec = 365 * 86400.0
        targets = sorted(save_years)
        targets_sec = [y * year_sec for y in targets]
        collected = []
        for t, pos, m, vel, E in sim_gen:
            while targets_sec and t >= targets_sec[0]:
                collected.append((t, pos.copy(), m.copy(), vel.copy(), E))
                targets_sec.pop(0)
            if not targets_sec:
                break

        if not collected:
            print("gif output failed!")
            return

        # Set up figure & update using first snapshot
        fig, update = setup_animation(collected[0])
        anim = FuncAnimation(fig, update,
                             frames=collected,
                             interval=200, blit=False,
                             repeat=False)

        # Save as GIF (uses matplotlib's PillowWriter backend)
        anim.save(gif_filename, writer='pillow', fps=10)
        print(f"Saved evolution GIF to '{gif_filename}'")
        return anim

    else:
        # Standard interactive animation
        # wrap sim_gen to decimate frames
        def decimated_gen():
            for i, frame in enumerate(sim_gen):
                if i % draw_interval[1] < draw_interval[2]:
                    yield frame
                elif i % draw_interval[0] == 0:
                    yield frame

        gen = decimated_gen()
        first = next(gen)
        fig, update = setup_animation(first)
        return FuncAnimation(fig, update,
                             frames=gen,
                             interval=50, blit=True,
                             cache_frame_data=False,
                             repeat=False)


if __name__ == '__main__':
    sim = simulate(seed=SEED)
    anim = animate(sim, draw_interval=DRAW_SKIP,
                   save_years=SAVE_YEARS,
                   gif_filename=GIF_FILENAME)
    # In interactive mode, SHOW the plot; in GIF mode, it's already saved
    if not SAVE_YEARS:
        plt.show()