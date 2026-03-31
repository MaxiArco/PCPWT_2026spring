import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


temps_c = np.array([0, 5, 10, 15, 20, 25, 30])
rhos_w = np.array([999.8, 1000.0, 999.7, 999.1, 998.2, 997.0, 995.7]) #  (kg/m^3)
mus_base = np.array([1.781, 1.518, 1.307, 1.139, 1.002, 0.890, 0.798])
mus = mus_base * 1e-3 # (Pa*s)


F_TITLE = 22       
F_LABEL = 20      
F_TICKS = 18       
F_LEGEND = 18      
L_WIDTH = 3.5      
A_ALPHA = 0.9      
def simulate_N_temp_lognormal(temp_c, mu, rho_w, total_steps=2000):
    T = temp_c + 273.15
    G = 25.0        #  (s^-1)
    alpha = 0.5     
    rho_p = 2650.0  # (kg/m^3)
    df = 2.3         
    d0 = 1e-6       #  (1 um)
    kb = 1.38e-23   
    
    n_bins = 30
    v0 = (np.pi/6) * d0**3
    v_bins = v0 * (1.5**np.arange(n_bins))
    d_bins = (6*v_bins/np.pi)**(1/3)
    
    # ---------------------------------------------------------
    # 2. Log-Normal Distribution
    # ---------------------------------------------------------
    mu_g = 2e-6   # (2 um)
    sigma_g = 1.8 # std
    s = np.log(sigma_g)
    
    unnormalized_N = lognorm.pdf(d_bins, s=s, scale=mu_g)
    
    ntu_g_L = 0.01
    target_N_total = ntu_g_L * 1000 / 6.28e-13
    
    N = unnormalized_N * (target_N_total / np.sum(unnormalized_N))
    

    beta = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            di, dj = d_bins[i], d_bins[j]
            b_br = (2*kb*T)/(3*mu) * (di+dj)**2 / (di*dj)
            b_sh = (1/6) * G * (di+dj)**3
            

            rho_floc_i = rho_w + (rho_p - rho_w) * (d0/di)**(3-df) if di > d0 else rho_p
            vi_settle = (9.81 * (rho_floc_i - rho_w) * di**2) / (18 * mu)
            
            rho_floc_j = rho_w + (rho_p - rho_w) * (d0/dj)**(3-df) if dj > d0 else rho_p
            vj_settle = (9.81 * (rho_floc_j - rho_w) * dj**2) / (18 * mu)
            
            b_ds = (np.pi/4) * (di+dj)**2 * abs(vi_settle - vj_settle)
            beta[i,j] = b_br + b_sh + b_ds


    dt = 1.0 
    N_total_history = []
    
    for step in range(total_steps):
        N_total_history.append(np.sum(N))
        
        dN = np.zeros(n_bins)
        
        for i in range(n_bins):
            for j in range(i, n_bins):
                # (i==j)  (i!=j)
                if i == j:
                    rate = 0.5 * alpha * beta[i,i] * (N[i]**2) * dt
                else:
                    rate = alpha * beta[i,j] * N[i] * N[j] * dt
                
                if rate <= 0: continue
                
                if i == j:
                    rate = min(rate, N[i] / 2.0)
                else:
                    rate = min(rate, min(N[i], N[j]))
                
                if rate <= 0: continue
                
                v_new = v_bins[i] + v_bins[j] 
                
                if v_new >= v_bins[-1]:
                    dN[i] -= rate * (2 if i==j else 1)
                    if i != j: dN[j] -= rate
                    dN[-1] += rate * (v_new / v_bins[-1])
                else:
                    k = np.searchsorted(v_bins, v_new, side='right') - 1
                    if k < n_bins - 1:
                        f = (v_bins[k+1] - v_new) / (v_bins[k+1] - v_bins[k])
                        dN[i] -= rate * (2 if i==j else 1)
                        if i != j: dN[j] -= rate
                        dN[k] += f * rate
                        dN[k+1] += (1 - f) * rate
                    else:
                        dN[i] -= rate * (2 if i==j else 1)
                        if i != j: dN[j] -= rate
                        dN[-1] += rate * (v_new / v_bins[-1])
                        
        N += dN
        N[N < 0] = 0 
        
    return N_total_history

results_N_lognorm = []
for i in range(len(temps_c)):
    results_N_lognorm.append(simulate_N_temp_lognormal(temps_c[i], mus[i], rhos_w[i], 2000))

plt.figure(figsize=(10, 6))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps_c)))

for i in range(len(temps_c)):
    plt.plot(results_N_lognorm[i], label=f'{temps_c[i]}°C', color=colors[i], linewidth=2.0)

plt.title("(b) Particle Concentration ($N_{total}$) Decay (Normal Distribution on 2$\mu m$)", fontsize=14, fontweight='bold')
plt.xlabel("Time Progression (Seconds)", fontsize=12)
plt.ylabel("Total Particle Concentration ($particles/m^3$)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Temperature", fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('coagulation_N_total_decay_lognormal.png', dpi=300)
plt.show()