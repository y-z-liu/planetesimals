import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 新增3D绘图模块
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from numba import njit, prange
# #this is a three dimensional version
# # ---------------- Tunable parameters ----------------
N_INIT         = 1000          # initial number of bodies
T_END_YEARS    = 100           # total evolution time in years

R_FACTOR       = 5             # collision radius scaling factor
TOT_MASS_RATIO = 10           # total mass divided by Earth mass

RHO_AU         = 0.1          # initial variation ratio of orbit
RHO_MASS       = 0.5           # initial variation ratio of mass
X_RATIO        = 0.1          # X-axis shrinking ratio
Y_RATIO        = 0.1           # Y-axis shrinking ratio
Z_THICK        = 0.01
COMPRESSION_FACTOR= 0.5        # compression due to collision
DENSITY        = 1           # initial density(divided by earth's density)

# adaptive time‐step bounds (seconds)
DT_MIN         = 1.0e2
DT_MAX         = 1.0e5
# visualization
COlOR          = "Blues"
INIT_SIZE      = 1
SCOPE_RATIO    = 2
Z_SCOPE_RATIO  = 0.2
VIEW           = False
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
    acc = np.zeros((n, 3), dtype=np.float64)  # 3-dimension
    for i in prange(n):
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]  # unpack three-dimensional coordinates
        ax, ay, az = 0.0, 0.0, 0.0

        # mutual gravity
        for j in range(n):
            if i == j:
                continue # ignore itself
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist_cubed = np.sqrt(dist_sq)**3  #  cubic power of distance
            if dist_cubed == 0:
                continue
            coeff = -G * masses[j] / dist_cubed
            ax += coeff * dx
            ay += coeff * dy
            az += coeff * dz
        
        # central star gravity
        r_sq = xi**2 + yi**2 + zi**2
        r_cubed = np.sqrt(r_sq)**3
        if r_cubed == 0:
            continue
        coeff = -G * M_STAR / r_cubed
        ax += coeff * xi
        ay += coeff * yi
        az += coeff * zi  
        
        acc[i] = [ax, ay, az]
    return acc

@njit
def leapfrog_step(pos, vel, masses, dt):
    """
    Verlet integral. 2-dim & 3-dim are both ok.
    """
    a0 = compute_accelerations(pos, masses)# return a 3-dim vector
    vel_half = vel + 0.5 * dt * a0
    pos_new = pos + dt * vel_half
    a1 = compute_accelerations(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * a1
    return pos_new, vel_new

@njit(parallel=True)
def neighbour_pairs(pos, cell_size):
    """
    Find candidate pairs whose grid cells are within one cell of each other,
    (in the  three dimensional space)
    """
    n = pos.shape[0]
    grid = np.empty((n,3), np.int64)  # grid coordinates
    
    for i in prange(n):
        grid[i,0] = int(pos[i,0] // cell_size)
        grid[i,1] = int(pos[i,1] // cell_size)
        grid[i,2] = int(pos[i,2] // cell_size)
    
    # count qualifying pairs
    cnt = 0
    for i in prange(n):
        for j in range(i+1, n):
            dx = abs(grid[i,0] - grid[j,0])
            dy = abs(grid[i,1] - grid[j,1])
            dz = abs(grid[i,2] - grid[j,2])
            if dx <=1 and dy <=1 and dz <=1:  # 三维邻域条件
                cnt +=1
    
    pairs = np.empty((cnt,2), dtype=np.int64)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(grid[i,0] - grid[j,0])
            dy = abs(grid[i,1] - grid[j,1])
            dz = abs(grid[i,2] - grid[j,2])
            if dx <=1 and dy <=1 and dz <=1:
                pairs[idx] = [i,j]
                idx +=1
    return pairs

@njit(parallel=True)
def min_time_to_collision(pos, vel, radii, pairs, R_factor):
    """
    For each candidate pair, solve |r0 + v_rel * t| = R_sum and return times.
    """
    n_pairs = pairs.shape[0]
    times = np.empty(n_pairs, dtype=np.float64)
    
    for k in prange(n_pairs):
        i, j = pairs[k]
        r0x = pos[i,0] - pos[j,0]
        r0y = pos[i,1] - pos[j,1]
        r0z = pos[i,2] - pos[j,2]  
        
        vrelx = vel[i,0] - vel[j,0]
        vrely = vel[i,1] - vel[j,1] 
        vrelz = vel[i,2] - vel[j,2]  
        
        R_sum = (radii[i] + radii[j]) * R_factor
        
        
        a = vrelx**2 + vrely**2 + vrelz**2
        b = 2.0 * (r0x*vrelx + r0y*vrely + r0z*vrelz)
        c = r0x**2 + r0y**2 + r0z**2 - R_sum**2

        # the following code logic is independent of the space dimension
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
    R_sq = R_star_scaled**2
    
    for i in prange(n):
        x, y, z = pos[i]
        vx, vy, vz = vel[i]
        
        # compute the nearest time
        v_dot_r = x*vx + y*vy + z*vz
        v_mag_sq = vx**2 + vy**2 + vz**2
        
        if v_mag_sq > 0:
            t_closest = max(0, min(-v_dot_r / v_mag_sq, dt))
        else:
            t_closest = 0
        
        
        x_t = x + vx * t_closest
        y_t = y + vy * t_closest 
        z_t = z + vz * t_closest  
        dist_sq = x_t**2 + y_t**2 + z_t**2
        
        if dist_sq <= R_sq:
            absorb[i] = True
    return absorb

@njit
def resolve_collisions_and_absorption(pos, vel, masses, radii, pairs, tcols, dt, R_factor, R_star_scaled):
    """
    Merge colliding bodies, remove star-absorbed ones, return updated arrays.
    """
    n = len(masses)
    parent = np.arange(n, dtype=np.int64)

    # Union-Find parent array
    def find_parent(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # union colliding pairs
    for k in range(len(pairs)):
        if tcols[k] <= dt:
            i, j = pairs[k]
            root_i = find_parent(i)
            root_j = find_parent(j)
            if root_i != root_j:
                parent[root_j] = root_i

    # find root for each body
    labels = np.array([find_parent(i) for i in range(n)])
    unique_labels = np.unique(labels)
    new_n = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}

    # allocate arrays for merged bodies
    new_pos = np.zeros((new_n, 3), dtype=np.float64)
    new_vel = np.zeros((new_n, 3), dtype=np.float64)
    new_mass = np.zeros(new_n)
    new_rad = np.zeros(new_n)
    
    for i in range(n):
        group = label_map[labels[i]]
        m = masses[i]
        new_mass[group] += m
        new_pos[group] += pos[i] * m
        new_vel[group] += vel[i] * m

    # finalize center of mass and compute radius
    for g in range(new_n):
        new_pos[g] /= new_mass[g]
        new_vel[g] /= new_mass[g]
        new_rad[g] = (new_mass[g] / (M_EARTH * DENSITY) )**(1/3.0) * R_EARTH * COMPRESSION_FACTOR #   consider compression due to collision, defaut = 1

    # detect and remove star-absorbed bodies
    absorb = detect_sun_absorption(new_pos, new_vel, dt, R_star_scaled)

    # allocate final arrays
    survivors = np.where(~absorb)[0]
    final_pos = new_pos[survivors]
    final_vel = new_vel[survivors]
    final_mass = new_mass[survivors]
    final_rad = new_rad[survivors]

    return final_pos, final_vel, final_mass, final_rad


def initialize_bodies(n, total_mass_ratio, speed_variation=0.0):
    """
    Initialize positions, velocities, masses, and radii of bodies.

    """
    mass_mean = total_mass_ratio * M_EARTH / n
    
    # 柱坐标
    r = AU * (1 + np.random.uniform(-RHO_AU, RHO_AU, n))
    theta = np.random.uniform(0, 2*np.pi, n)
    z = AU * Z_THICK * np.random.uniform(-1, 1, n)  # z方向厚度为0.2 AU
    
    # Decartes coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pos = np.vstack([x, y, z]).T
    
    # initial velocities(spin in xOy& desturbance along z)
    v_circ = np.sqrt(G * M_STAR / r)# (设为圆周运动的稳定速度)
    v_theta = v_circ * (1 + np.random.uniform(-speed_variation, speed_variation, n))
    vz = 0.001 * v_circ * np.random.uniform(-1, 1, n)  # 垂直方向速度
    
    vel = np.vstack([
        -v_theta * np.sin(theta),
         v_theta * np.cos(theta),
         vz
    ]).T
    
    masses = mass_mean + np.random.uniform(-RHO_MASS, RHO_MASS, n)
    radii  = (masses / (M_EARTH * DENSITY))**(1/3) * R_EARTH
    
    return pos, vel, masses, radii

def simulate(init_bodies, n=N_INIT, total_mass_ratio=TOT_MASS_RATIO, t_end_years=T_END_YEARS):
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
        vel_mags = np.linalg.norm(vel, axis=1)
        max_speed = np.max(vel_mags)
        cell_size = 2*(np.max(radii)*R_FACTOR + max_speed* dt)
        
        pairs = neighbour_pairs(pos, cell_size)

        # compute collision times for all candidate pairs
        if len(pairs) > 0:
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
        pos, vel, masses, radii = resolve_collisions_and_absorption(
            pos, vel, masses, radii,
            pairs, tcols, dt, R_FACTOR, R_STAR*R_FACTOR
        )

        t += dt
        if (t % 86400 * 365) ==0:
            print(f"Time: {t/86400/365:.2f} yr | Bodies: {len(masses)} | dt: {dt:.1e} s")
        yield t, np.copy(pos), np.copy(masses)

        # update dt for next iteration

        dt = dt_next






def animate_3d(sim_gen):
    """
    Visualize the simulation using matplotlib animation.
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')  # 3D projection
    
    # central star
    ax.scatter([0], [0], [0], s=100, c='red', marker='.')
    
    # 坐标轴设置
    ax.set_xlim3d(-SCOPE_RATIO*AU, SCOPE_RATIO*AU)
    ax.set_ylim3d(-SCOPE_RATIO*AU, SCOPE_RATIO*AU)
    ax.set_zlim3d(-Z_SCOPE_RATIO * AU, Z_SCOPE_RATIO * AU)  # z轴范围较小体现盘状结构
    ax.set_box_aspect([2,2,0.5])  # 空间比例
    

    mass_mean = TOT_MASS_RATIO * M_EARTH / N_INIT
    norm = LogNorm(vmin=mass_mean*0.1, vmax=mass_mean*0.5*N_INIT)
    s_init = INIT_SIZE
    
    scat = ax.scatter([], [], [], s=s_init, c=[], 
                     cmap=COlOR, norm=norm, depthshade=True)
    
    def update(frame):
        t, pos, masses = frame
        
        
        s_current = np.power(masses / mass_mean, 2.0/3.0) * INIT_SIZE
        
        # update scatter attributes
        scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])  
        scat.set_array(masses)
        scat.set_sizes(s_current)
        
        # spining view
        if VIEW:
            ax.view_init(elev=25, azim=0.3*t/86400)  
        
        ax.set_title(f"3D View | t = {t/86400/365:.1f} yr | N = {len(masses)}")
        return scat
    
    return FuncAnimation(fig, update, frames=sim_gen,
                        interval=20, blit=False, 
                        cache_frame_data=False)


init_bodies = initialize_bodies(N_INIT, TOT_MASS_RATIO)
sim_gen = simulate(init_bodies)
anim = animate_3d(sim_gen)  
plt.show()
