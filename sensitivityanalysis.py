import numpy as np
import matplotlib.pyplot as plt


F_TITLE = 22       
F_LABEL = 20       
F_TICKS = 18       
F_LEGEND = 18      
L_WIDTH = 3.5      
A_ALPHA = 0.9      

g = 9.81
rho_p = 2650.0  
d0 = 1e-6       # (1 um)
d_array = np.logspace(-6, -3, 500) 

# (0, 10, 20, 30 °C)
temps = [0, 10, 20, 30]
rhos_w = [999.8, 999.7, 998.2, 995.7]
mus = [1.781e-3, 1.307e-3, 1.002e-3, 0.798e-3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# ==============================================================================
# (a) (df = 2.3)
# ==============================================================================
df_fixed = 2.3
colors_t = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728']

axins = ax1.inset_axes([0.68, 0.08, 0.28, 0.35]) 

for T, rho_w, mu, color in zip(temps, rhos_w, mus, colors_t):
    rho_floc = rho_w + (rho_p - rho_w) * (d0 / d_array)**(3 - df_fixed)
    vs_ms = (g * (rho_floc - rho_w) * d_array**2) / (18 * mu)
    vs_mh = vs_ms * 3600 
    
    ax1.plot(d_array * 1e6, vs_mh, label=f'{T}°C', color=color, lw=L_WIDTH, alpha=A_ALPHA)
    axins.plot(d_array * 1e6, vs_mh, color=color, lw=L_WIDTH, alpha=A_ALPHA)

ax1.set_title(f'(a) Effect of Temperature ($d_f = {df_fixed}$)', fontsize=F_TITLE, fontweight='bold', pad=20)
ax1.set_xlabel('Particle Diameter ($\mu m$)', fontsize=F_LABEL)
ax1.set_ylabel('Settling Velocity ($m/h$)', fontsize=F_LABEL)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='both', which='major', labelsize=F_TICKS)
ax1.tick_params(axis='both', which='minor', labelsize=F_TICKS)
ax1.legend(title='Water Temp.', fontsize=F_LEGEND, title_fontsize=F_LEGEND, loc='upper left', framealpha=0.8)



axins.set_xlim(80, 120)
axins.set_ylim(0.5, 2.0)
axins.set_xscale('linear')
axins.set_yscale('linear')

axins.tick_params(labelsize=12)


axins.axvline(x=100, color='gray', linestyle='--', lw=2, alpha=0.7)


rect, connects = ax1.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, lw=2)


for c in connects:
    c.set_visible(False)


# ==============================================================================
# (b) (T = 20°C)
# ==============================================================================
dfs = [2.0, 2.3, 2.6, 3.0]
labels_df = ['2.0 (Loose)', '2.3 (Typical)', '2.6 (Compact)', '3.0 (Solid Sphere)']
colors_df = ['#2ca02c', '#ff7f0e', '#8c564b', '#7f7f7f']

T_fixed_idx = 2 
rho_w_fixed = rhos_w[T_fixed_idx]
mu_fixed = mus[T_fixed_idx]

for df_val, label, color in zip(dfs, labels_df, colors_df):
    if df_val == 3.0:
        rho_floc = np.full_like(d_array, rho_p) 
    else:
        rho_floc = rho_w_fixed + (rho_p - rho_w_fixed) * (d0 / d_array)**(3 - df_val)
        
    vs_ms = (g * (rho_floc - rho_w_fixed) * d_array**2) / (18 * mu_fixed)
    vs_mh = vs_ms * 3600
    
    line_style = '--' if df_val == 3.0 else '-'
    ax2.plot(d_array * 1e6, vs_mh, label=f'{label}', color=color, ls=line_style, lw=L_WIDTH, alpha=A_ALPHA)

ax2.set_title('(b) Effect of Floc Structure ($T = 20^\circ C$)', fontsize=F_TITLE, fontweight='bold', pad=20)
ax2.set_xlabel('Particle Diameter ($\mu m$)', fontsize=F_LABEL)
ax2.set_ylabel('Settling Velocity ($m/h$)', fontsize=F_LABEL)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(axis='both', which='major', labelsize=F_TICKS)
ax2.tick_params(axis='both', which='minor', labelsize=F_TICKS)
ax2.legend(title='Fractal Dim. ($d_f$)', fontsize=F_LEGEND, title_fontsize=F_LEGEND, loc='upper left', framealpha=0.8)
ax2.set_ylim(ax1.get_ylim())

plt.tight_layout()
ax = plt.gca()
for ax in [ax1, ax2, axins]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.subplots_adjust(hspace=0.25)
output_filename = 'Settling_Velocity_Stacked_With_Inset_Adjusted.png'
plt.savefig(output_filename, dpi=300)

plt.show()