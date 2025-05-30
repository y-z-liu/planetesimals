import os
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

# ---------------- Tunable parameters ----------------
N_INIT          = 1000                # initial number of micro-planets
T_END_YEARS     = 10000               # total integration time [years]
DRAW_SKIP       = [1e3, 1e6, 1e3]     # draw intervals for interactive animation

R_FACTOR        = 2                   # collision-radius scaling factor
TOT_MASS_RATIO  = 5                   # total planetary mass (Earth masses)

RHO_AU          = 0.02                # orbital scatter fraction
RHO_MASS        = 0.2                 # mass scatter fraction

DT_MIN          = 1.0e2               # minimum timestep [s]
DT_MAX          = 1.0e5               # maximum timestep [s]

COLOR           = "Blues"             # colormap for mass (2D mode)
PLOT_SCALE      = 2e-6                # planet scatter marker size scale
PLT_STYLE       = "Solarize_Light2"   # matplotlib style

SEED            = 2025                # random seed (None to disable)
SAVE_YEARS      = list(np.arange(100)/50) \
                  + list(range(2,100,2)) \
                  + list(range(100,10000,200)) \
                  + list(np.arange(499900,500002)/50)
GIF_FILENAME    = 'selected_frames.gif'
GIF_FPS         = 10

SAVE_GIF        = False

# State load/save parameters
LOAD_STATE      = False
SAVE_STATE      = True
STATE_FILENAME  = 'saved_state.npz'

# 3D simulation flag and z-scatter
THREE_D         = True
RHO_Z           = 0.0001
COLOR_FACTOR    = 0.01               # how much AU to z=0 to be colored red/green.
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
    """Compute gravitational acceleration in 3D (works for 2D case if z=0)."""
    n = pos.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64)
    for i in prange(n):
        xi, yi, zi = pos[i]
        axi = ayi = azi = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            dz = zi - pos[j,2]
            inv_r3 = 1.0 / (dx*dx + dy*dy + dz*dz)**1.5
            axi += -G * masses[j] * dx * inv_r3
            ayi += -G * masses[j] * dy * inv_r3
            azi += -G * masses[j] * dz * inv_r3
        acc[i,0], acc[i,1], acc[i,2] = axi, ayi, azi
    return acc

@njit
def leapfrog_step(pos, vel, masses, dt):
    """Perform one leapfrog integration step."""
    a0 = compute_accelerations(pos, masses)
    vel_half = vel + 0.5 * dt * a0
    pos_new = pos + dt * vel_half
    a1 = compute_accelerations(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * a1
    return pos_new, vel_new

@njit(parallel=True)
def neighbour_pairs(pos, cell_size):
    """Spatial hashing in x,y only (z ignored) to find candidate collision pairs."""
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

    pairs = np.empty((cnt, 2), dtype=np.int64)
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
    """Compute soonest collision times for each candidate pair in 3D."""
    m = pairs.shape[0]
    times = np.empty(m, dtype=np.float64)
    for k in prange(m):
        i, j = pairs[k]
        dx = pos[i,0] - pos[j,0]
        dy = pos[i,1] - pos[j,1]
        dz = pos[i,2] - pos[j,2]
        dvx = vel[i,0] - vel[j,0]
        dvy = vel[i,1] - vel[j,1]
        dvz = vel[i,2] - vel[j,2]
        Rsum = (radii[i] + radii[j]) * R_factor

        a = dvx*dvx + dvy*dvy + dvz*dvz
        b = 2.0 * (dx*dvx + dy*dvy + dz*dvz)
        c = dx*dx + dy*dy + dz*dz - Rsum*Rsum

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
    """Merge bodies that collide within dt, mass-weighted in 3D."""
    n = masses.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Union-find to group colliding bodies
    for k in range(pairs.shape[0]):
        i, j = pairs[k]
        if tcols[k] <= dt:
            pi, pj = find(i), find(j)
            if pi != pj:
                if pi == 0 or pj == 0:
                    parent[max(pi, pj)] = 0
                else:
                    parent[pj] = pi

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

    new_pos  = np.zeros((new_n, 3), dtype=np.float64)
    new_vel  = np.zeros((new_n, 3), dtype=np.float64)
    new_mass = np.zeros(new_n, dtype=np.float64)
    new_rad  = np.zeros(new_n, dtype=np.float64)

    # accumulate mass, momentum, and center-of-mass position
    for i in range(n):
        g = idx_map[labels[i]]
        m = masses[i]
        new_mass[g] += m
        new_pos[g]  += pos[i] * m
        new_vel[g]  += vel[i] * m

    # finalize center-of-mass and compute radii
    star_group = idx_map[find(0)]
    for g in range(new_n):
        mt = new_mass[g]
        new_pos[g] /= mt
        new_vel[g] /= mt
        new_rad[g] = R_STAR if g == star_group else (mt / M_EARTH)**(1/3) * R_EARTH

    return new_pos, new_vel, new_mass, new_rad

def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    """Randomly place n bodies around the star, optionally in 3D."""
    mass_mean = total_mass_ratio * M_EARTH / n

    # radial distance with small scatter in the orbital plane
    r0    = AU + np.random.uniform(-RHO_AU*AU, RHO_AU*AU, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    pos_xy = np.vstack([r0*np.cos(theta), r0*np.sin(theta)]).T

    # circular speed ± variation
    v_circ = np.sqrt(G * M_STAR / r0)
    speed  = v_circ * (1 + np.random.uniform(-speed_variation, speed_variation, n))
    vel_xy = np.vstack([-speed*np.sin(theta), speed*np.cos(theta)]).T

    # masses & radii
    masses_p = mass_mean * (1 + np.random.uniform(-RHO_MASS, RHO_MASS, n))
    radii_p  = (masses_p / M_EARTH)**(1/3) * R_EARTH

    # decide initial z‐coordinate
    if THREE_D:
        z0 = np.random.uniform(-RHO_Z*AU, RHO_Z*AU, n)
    else:
        z0 = np.zeros(n)

    pos_p = np.column_stack((pos_xy, z0))
    vel_p = np.column_stack((vel_xy, np.zeros(n)))

    # prepend the central star at the origin
    pos    = np.vstack((np.zeros((1,3)), pos_p))
    vel    = np.vstack((np.zeros((1,3)), vel_p))
    masses = np.concatenate((np.array([M_STAR]), masses_p))
    radii  = np.concatenate((np.array([R_STAR]),  radii_p))

    return pos, vel, masses, radii

@njit(parallel=True)
def compute_total_energy(pos, vel, masses):
    """Compute total mechanical energy in 3D."""
    n = pos.shape[0]
    KE = 0.0
    for i in range(n):
        KE += 0.5 * masses[i] * np.dot(vel[i], vel[i])
    PE = 0.0
    for i in prange(n):
        mi = masses[i]
        xi, yi, zi = pos[i]
        for j in range(i+1, n):
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            dz = zi - pos[j,2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            PE += -G * mi * masses[j] / dist
    return KE + PE

def simulate(n=N_INIT,
             total_mass_ratio=TOT_MASS_RATIO,
             t_end_years=T_END_YEARS,
             seed=SEED,
             load_state=LOAD_STATE,
             save_state=SAVE_STATE,
             state_filename=STATE_FILENAME):
    """
    Yields (t, pos, masses, vel, energy) frames, with optional load/save.
    """
    if seed is not None:
        np.random.seed(seed)

    if load_state and os.path.exists(state_filename):
        data   = np.load(state_filename)
        t      = float(data['t'])
        pos    = data['pos']
        vel    = data['vel']
        masses = data['masses']
        radii  = data['radii']
        print(f"Loaded state from '{state_filename}' at t={t/(365*86400):.2f} yr")
    else:
        if load_state:
            print(f"No state file found; starting fresh")
        pos, vel, masses, radii = initialize_bodies(n, total_mass_ratio)
        t = 0.0

    t_end = t_end_years * 365 * 86400.0
    dt = DT_MIN

    while t < t_end and masses.size > 1:
        max_v = np.max(np.linalg.norm(vel, axis=1))
        cell  = 2 * np.max(radii) * R_FACTOR + 2 * max_v * dt

        pairs = neighbour_pairs(pos, cell)
        tcols = (min_time_to_collision(pos, vel, radii, pairs, R_FACTOR)
                 if pairs.size else np.empty(0))

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

    if save_state:
        np.savez(state_filename,
                 t=t, pos=pos, vel=vel,
                 masses=masses, radii=radii)
        print(f"Saved final state at t={t/(365*86400):.2f} yr")

def animate(sim_gen,
            draw_interval=DRAW_SKIP,
            save_years=SAVE_YEARS,
            gif_filename=GIF_FILENAME,
            three_d=THREE_D,
            save_gif=SAVE_GIF):
    """
    If save_gif is True, captures snapshots into a GIF (at years in save_years);
    otherwise launches an interactive (3D-aware) real-time animation.
    """
    def setup_animation(first_frame):
        # Unpack first frame
        t0, pos0, m0, _, E0 = first_frame
        star_x, star_y, _ = pos0[0]
        pos_p0 = pos0[1:]; m_p0 = m0[1:]

        # Styling and figure layout
        plt.style.use(PLT_STYLE)
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2,
                              width_ratios=[2, 1],
                              height_ratios=[3, 3, 3],
                              wspace=0.3, hspace=0.7)

        # Main scatter plot
        ax_sc = fig.add_subplot(gs[:, 0])
        ax_sc.set_xlim(-2.5*AU, 2.5*AU)
        ax_sc.set_ylim(-2.5*AU, 2.5*AU)
        ax_sc.set_aspect('equal')
        ax_sc.xaxis.set_major_locator(plt.MultipleLocator(AU))
        ax_sc.yaxis.set_major_locator(plt.MultipleLocator(AU))
        ax_sc.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/AU:.0f} AU'))
        ax_sc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/AU:.0f} AU'))
        star_marker = ax_sc.scatter([star_x], [star_y],
                                    s=300, marker=(12,1,0), c='orange')

        sizes = ((m_p0 / M_EARTH)**(1/3) * R_EARTH * PLOT_SCALE)**2

        if three_d:
            norm_z = Normalize(vmin=-AU*COLOR_FACTOR, vmax=AU*COLOR_FACTOR)
            alpha = 0.6
            z_cmap = LinearSegmentedColormap.from_list(
                'zmap', [
                    (0.0, (0.0, 1.0, 0.0, alpha)),
                    (0.5, (0.0, 0.0, 1.0, alpha)),
                    (1.0, (1.0, 0.0, 0.0, alpha)), 
                ]
            )
            scat = ax_sc.scatter(
                pos_p0[:,0], pos_p0[:,1],
                c=pos_p0[:,2],
                cmap=z_cmap,
                norm=norm_z,
                s=sizes,
                edgecolors='none'
            )
        else:
            mean_mass = TOT_MASS_RATIO * M_EARTH / N_INIT
            norm_m = LogNorm(vmin=mean_mass*0.1,
                             vmax=mean_mass*N_INIT)
            scat = ax_sc.scatter(pos_p0[:, 0], pos_p0[:, 1],
                                 c=m_p0, cmap=COLOR,
                                 norm=norm_m, s=sizes)

        # Mass distribution histogram
        ax_h = fig.add_subplot(gs[0,1])
        min_em = (1 - RHO_MASS) * TOT_MASS_RATIO / N_INIT
        max_em = (1 + RHO_MASS) * TOT_MASS_RATIO
        bins   = np.logspace(np.log10(min_em), np.log10(max_em), 30)
        cnt0, _ = np.histogram(m_p0/M_EARTH, bins=bins)
        cnt0 = np.where(cnt0>0, cnt0, 1)
        bars = ax_h.bar(bins[:-1], cnt0,
                        width=np.diff(bins),
                        align='edge', edgecolor='black')
        ax_h.set_xscale('log')
        ax_h.set_yscale('log')
        ax_h.set_xlabel('M (Earth masses)')
        ax_h.set_ylabel('Count')
        ax_h.set_title('Mass Distribution')
        ax_h.set_ylim(1e-1, cnt0.max()*1.2)

        # Number of bodies vs time
        ax_n = fig.add_subplot(gs[1,1])
        times  = [t0/(86400*365)]
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
            t, pos, m, _, E = frame
            star_marker.set_offsets(pos[0,:2])
            pos_p = pos[1:]; m_p = m[1:]
            sizes = ((m_p/M_EARTH)**(1/3)*R_EARTH * PLOT_SCALE)**2

            if three_d:
                zs = pos_p[:,2]
                scat.set_offsets(pos_p[:,:2])
                scat.set_array(zs)
                scat.set_sizes(sizes)
            else:
                scat.set_offsets(pos_p[:,:2])
                scat.set_array(m_p)
                scat.set_sizes(sizes)

            ax_sc.set_title(f't = {t/(86400*365):.2f} yr   N = {m_p.size}')

            cnt, _ = np.histogram(m_p/M_EARTH, bins=bins)
            cnt = np.where(cnt>0, cnt, 1e-8)
            for rect, h in zip(bars, cnt):
                rect.set_height(h)
            ax_h.set_ylim(1e-1, cnt.max()*1.2)

            times.append(t/(86400*365))
            counts.append(m_p.size)
            line_count.set_data(times, counts)
            ax_n.set_xlim(0, times[-1]*1.05)
            ax_n.set_ylim(0, max(counts)*1.05)

            energies.append(E)
            line_energy.set_data(times, energies)
            ax_e.set_xlim(0, times[-1]*1.05)
            ax_e.set_ylim(min(energies), max(energies))

            return (star_marker, scat, *bars, line_count, line_energy)

        return fig, update

    # Decide GIF vs interactive mode
    if save_gif:
        # Collect frames at specified years for GIF
        year_sec = 365 * 86400.0
        targets = sorted(save_years)
        ts = [y * year_sec for y in targets]
        collected = []
        for frame in sim_gen:
            t, _, m, _, _ = frame
            while ts and t >= ts[0]:
                collected.append(frame)
                print("collected t=", t/year_sec, "yrs, N=", len(m)-1)
                ts.pop(0)
            if not ts:
                break
        if not collected:
            print("No frames collected for GIF.")
            return
        fig, update = setup_animation(collected[0])
        anim = FuncAnimation(fig, update,
                             frames=collected,
                             interval=200,
                             blit=False,
                             repeat=False)
        anim.save(gif_filename, writer='pillow', fps=GIF_FPS)
        print("finishing... seed=", SEED, " R=", R_FACTOR," M=", TOT_MASS_RATIO, " 3D=", THREE_D)
        print(f"Saved evolution GIF to '{gif_filename}'")
        return anim

    else:
        # Interactive real-time decimated animation
        def decimated():
            for i, f in enumerate(sim_gen):
                if i % draw_interval[1] < draw_interval[2] or i % draw_interval[0] == 0:
                    yield f
        gen = decimated()
        first = next(gen)
        fig, update = setup_animation(first)
        return FuncAnimation(fig,
                             update,
                             frames=gen,
                             interval=50,
                             blit=False,
                             cache_frame_data=False,
                             repeat=False)

if __name__ == '__main__':
    sim = simulate(seed=SEED,
                   load_state=LOAD_STATE,
                   save_state=SAVE_STATE,
                   state_filename=STATE_FILENAME)
    anim = animate(sim,
                   draw_interval=DRAW_SKIP,
                   save_years=SAVE_YEARS,
                   gif_filename=GIF_FILENAME,
                   three_d=THREE_D,
                   save_gif=SAVE_GIF)
    if not SAVE_GIF:
        plt.show()