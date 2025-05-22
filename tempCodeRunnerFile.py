        if VIEW:
            ax.view_init(elev=25, azim=0.3*t/86400)

        ax.set_title(f"3D View | t = {t/86400/365:.1f} yr | N = {len(masses)}")