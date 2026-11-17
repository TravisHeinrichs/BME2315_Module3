#task 3, part 1: cavalitidePKPD.py = PK model from Task 2 + receptor binding (PD) model from Task 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

drug = 'Cavalitide'

# product-specific parameters
products = {
    'Ozempic': dict(Favailable=0.89,   # bioavailability
                    ka=0.029,          # 1/h, absorption rate
                    D=2.0,             # mg per dose
                    interval=168,      # h between doses (weekly)
                    n_doses=8,         # number of doses
                    depot='subcutaneous'),
    'Wegovy':  dict(Favailable=0.01,
                    ka=0.0029,
                    D=25,               # mg
                    interval=24,       # h between doses (daily)
                    n_doses=100,
                    depot='gut (oral)'),
    'Cavalitide': dict(
        Favailable = 0.01,
        ka = 0.006,
        D = 20,            # mg, Phase 1 dose from Task 2 (gives ~0.5 nM free in blood)
        interval = 24,     # daily pill
        n_doses = 100,     # 100 daily doses = first 100 days of treatment
        depot='gut (oral)')
}

p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of drug bound to albumin

# simulation parameters
dt = 0.1        # h, time step between output points

# receptor binding (PD) parameters
Kd1 = 0.17      # [nM] GLP1
Kd2 = 0.077     # [nM] Cavalitide, our fitted Kd from Task 1
L1 = 0.015      # [nM] GLP1 (fasting), same value used in Task 1

def ODEfunc(t, y):
    Asc, Cc = y
    dAsc = -ka*Asc            # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
    return [dAsc, dCc]

def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))      # time steps per dosing interval
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        sol = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y, t_eval = t_seg)  # solve the ODEs until the next dose
        y_seg = sol.y
        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy()         # carry state into next interval
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 (GLP1) and L2 (Cavalitide),
    same model as find_F_act in Task 1"""
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# --- run PK model ---
t, Asc, Cc = simulate(D, n_doses, interval, dt)

# --- PD model: only free Cavalitide (not bound to albumin) can bind the GLP1 receptor ---
L2 = Ffree*Cc                       # [nM] free Cavalitide in blood over time
Fact_t = Fact(L1, L2, Kd1, Kd2)     # fraction of active GLP1R over time

# --- plot ---
plt.figure()
plt.plot(t / 24, L2, label="free in blood [nM]")
plt.plot(t / 24, Fact_t, label="fraction active GLP1R (Fact)")
plt.axhline(Fact(L1, 0, Kd1, Kd2), ls=":", c="gray", lw=1, label="Fact with GLP1 alone (no drug)")
plt.ylim(0, 1)
plt.xlabel("Time (days)")
plt.ylabel("Free Cavalitide [nM] and Fact")
plt.title(f"{drug} PKPD: {D} mg every {interval} h, {n_doses} doses")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Fact with GLP1 alone (day 0) = {Fact(L1, 0, Kd1, Kd2):.3f}")
print(f"First day with Fact above 0.5 = day {t[Fact_t > 0.5][0]/24:.1f}")   # first time point with over half the receptors active
print(f"Free Cavalitide at day 100 = {L2[-1]:.3f} nM")
print(f"Fact at day 100 = {Fact_t[-1]:.3f}")


#task 3, part 2: find the daily dose that gives ~90% active GLP1R at steady state (day 100)
Fact_target = 0.90      # goal: 90% of GLP1 receptors active

# Brute force optimization of the dose D: try many candidate doses in a for loop
# (same method as Approach 1 in fit_semaglutide_GLP1R.py)
D_candidates = np.arange(20, 41, 1)   # [mg] candidate daily doses, 20 to 40 mg
SE_brute = np.inf
for D_try in D_candidates:
    t, Asc, Cc = simulate(D_try, n_doses, interval, dt)
    Fact_ss = Fact(L1, Ffree*Cc[-1], Kd1, Kd2)    # Fact at day 100 (steady state)
    SE = (Fact_ss - Fact_target)**2               # squared error from the 90% goal
    if SE < SE_brute:
        SE_brute = SE
        D_brute = D_try
print(f"Brute-force dose = {D_brute} mg every {interval} h (squared error = {SE_brute:.6f})")

# --- compare the Phase 1 dose (20 mg) with the optimized dose ---
plt.figure()
for D_plot in [20, D_brute]:
    t, Asc, Cc = simulate(D_plot, n_doses, interval, dt)
    Fact_t = Fact(L1, Ffree*Cc, Kd1, Kd2)
    plt.plot(t / 24, Fact_t, lw=2, label=f"{D_plot} mg every {interval} h")
    print(f"{D_plot} mg: free Cavalitide at day 100 = {Ffree*Cc[-1]:.3f} nM, Fact at day 100 = {Fact_t[-1]:.3f}")
plt.axhline(Fact_target, ls="--", c="gray", lw=1, label="90% target")
plt.axhline(Fact(L1, 0, Kd1, Kd2), ls=":", c="gray", lw=1, label="GLP1 alone (no drug)")
plt.ylim(0, 1)
plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title(f"{drug} PKPD: daily dose optimized for 90% active GLP1R")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#task 3, comparison: GLP1R activation for Cavalitide vs. semaglutide (Ozempic and Wegovy) over 100 days
Kd_drugs = {'Ozempic': 0.35,      # [nM] semaglutide Kd (value used in Task 1)
            'Wegovy': 0.35,       # [nM] semaglutide Kd
            'Cavalitide': 0.077}  # [nM] Cavalitide Kd fitted in Task 1
products['Ozempic']['n_doses'] = 15       # 15 weekly shots so Ozempic also covers 100 days
products['Cavalitide']['D'] = D_brute     # optimized Cavalitide dose from part 2

plt.figure()
for drug in ['Ozempic', 'Wegovy', 'Cavalitide']:
    p = products[drug]
    Favailable, ka, D = p['Favailable'], p['ka'], p['D']
    interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']
    t, Asc, Cc = simulate(D, n_doses, interval, dt)
    Fact_t = Fact(L1, Ffree*Cc, Kd1, Kd_drugs[drug])     # same receptor model, each drug's own Kd
    plt.plot(t / 24, Fact_t, lw=2, label=f"{drug}, {depot}: {D} mg every {interval} h")
    last_week = (t >= 93*24) & (t <= 100*24)              # time points in days 93 to 100
    print(f"{drug}: average Fact in days 93-100 = {np.mean(Fact_t[last_week]):.3f}")
plt.axhline(0.90, ls="--", c="gray", lw=1, label="90% target")
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1R activation: Cavalitide vs. semaglutide")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

