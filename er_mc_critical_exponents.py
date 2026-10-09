import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings


warnings.filterwarnings("ignore")


def set_nature_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
    markers = ['o', 's', '^', 'D']
    return colors, markers


def precompute_poisson(c, k_max):

    k_vals = np.arange(0, k_max + 1)
    P_k = stats.poisson.pmf(k_vals, c)
    return P_k


def calculate_theoretical_pc_pos_fix(R, c):

    kappa = c + 1.0
    low = 0.0
    high = 1.0

    for _ in range(1000):
        p = (low + high) / 2.0
        term_x = (1.0 - p) * kappa

        if abs(term_x - 1.0) < 1e-9:
            fraction = float(R)
        else:
            fraction = (term_x ** R - 1.0) / (term_x - 1.0)

        lhs = p * kappa * fraction

        if lhs < 1.0:
            low = p
        else:
            high = p
        if (high - low) < 1e-9:
            break

    return (low + high) / 2.0


def run_iteration_pos_fix(p, R, c, P_k):

    S = np.full(R, 0.1)
    S_new = np.zeros(R)

    max_iter = 50000
    err_max = 1e-9
    steps = 0
    err = 1.0


    def calc_term_val(theta, c_val):
        x = 1.0 - theta
        return x * np.exp(c_val * (x - 1.0))

    while err > err_max and steps < max_iter:
        S_old = S.copy()


        theta_R = np.sum(S_old)
        term1_sum = 1.0 - calc_term_val(theta_R, c)
        S_new[0] = p * term1_sum


        for r in range(1, R):
            theta_prev = np.sum(S_old[:r - 1])
            theta_curr = np.sum(S_old[:r])
            term_diff = calc_term_val(theta_prev, c) - calc_term_val(theta_curr, c)
            S_new[r] = (1.0 - p) * term_diff

        err = np.sqrt(np.sum((S - S_new) ** 2))
        S[:] = S_new[:]
        steps += 1


    sum_S_R = np.sum(S)
    sum_S_R_minus_1 = np.sum(S[:-1])
    k_indices = np.arange(len(P_k))


    term_S_p = np.sum(P_k * (1.0 - np.power(1.0 - sum_S_R, k_indices)))
    term_U_p = np.sum(P_k * (1.0 - np.power(1.0 - sum_S_R_minus_1, k_indices)))

    S_plus = p * term_S_p
    U_plus = (1.0 - p) * term_U_p


    term_SCGC_T = np.sum(P_k * np.power(1.0 - np.power(1.0 - sum_S_R, k_indices), 2))
    term_SCGC_UT = np.sum(P_k * np.power(1.0 - np.power(1.0 - sum_S_R_minus_1, k_indices), 2))

    return S_plus, U_plus, term_SCGC_T, term_SCGC_UT


def collect_data_log_space(pc, R, c, P_k):
    print(f"正在采集数据 (pc={pc:.6f})...")


    log_diffs = np.linspace(-2.5, -0.6, 50)


    p_values = pc + np.power(10, log_diffs)

    data = {'log_diff': [], 'S': [], 'U_plus': [], 'R': [], 'U': []}

    for i, p in enumerate(p_values):
        if p >= 1.0: continue

        S, U_p, R_v, U_t = run_iteration_pos_fix(p, R, c, P_k)

        if S > 1e-10:
            data['log_diff'].append(log_diffs[i])

            data['S'].append(np.log10(S))
            data['U_plus'].append(np.log10(U_p) if U_p > 0 else -10)

            val_R = R_v
            val_U = U_t

            data['R'].append(np.log10(val_R) if val_R > 0 else -10)
            data['U'].append(np.log10(val_U) if val_U > 0 else -10)

    return data


def main():
    R = 2
    c = 2.5
    k_max = 100

    print(f"Network: ER (Positive Correlation), c={c}, R={R}")

    P_k = precompute_poisson(c, k_max)
    pc_exact = calculate_theoretical_pc_pos_fix(R, c)
    print(f"精确理论 pc = {pc_exact:.8f}")

    raw_data = collect_data_log_space(pc_exact, R, c, P_k)


    filename_txt = f"er_pos_critical_data_c{c}_R{R}.txt"
    with open(filename_txt, "w") as f:
        f.write(f"# ER Positive Correlation Network Data (Standard Range)\n")
        f.write(f"# c={c}, R={R}, pc_exact={pc_exact:.8f}\n")
        f.write(f"# Columns: log10(p-pc)\tlog10(S)\tlog10(U_plus)\tlog10(R)\tlog10(U)\n")
        for i in range(len(raw_data['log_diff'])):
            line = (f"{raw_data['log_diff'][i]:.6f}\t"
                    f"{raw_data['S'][i]:.6f}\t"
                    f"{raw_data['U_plus'][i]:.6f}\t"
                    f"{raw_data['R'][i]:.6f}\t"
                    f"{raw_data['U'][i]:.6f}\n")
            f.write(line)

    plot_results(raw_data, R, c)


def plot_results(raw_data, R, c):
    colors, markers = set_nature_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    keys = ['S', 'U_plus', 'R', 'U']
    labels = [r'$S$', r'$U^+$', r'$R$', r'$U$']


    FIT_POINTS = 20

    print("\n拟合结果 (ER POS, Standard Sampling, Fitting First 20):")

    for i, key in enumerate(keys):
        x_data = np.array(raw_data['log_diff'])
        y_data = np.array(raw_data[key])

        if len(x_data) > 5:

            points_to_fit = min(len(x_data), FIT_POINTS)
            slope, intercept, r_value, _, _ = stats.linregress(
                x_data[:points_to_fit],
                y_data[:points_to_fit]
            )
            print(f"  {key}: Beta={slope:.4f} (R2={r_value ** 2:.4f})")


            ax.scatter(x_data, y_data, color=colors[i], marker=markers[i],
                       s=25, alpha=0.3, edgecolors='none')


            x_line = np.linspace(min(x_data), max(x_data), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=colors[i], linewidth=1.5, linestyle='--',
                    label=f'{labels[i]} ($\\beta={slope:.2f}$)')


            ax.scatter(x_data[::4], y_data[::4], color=colors[i], marker=markers[i],
                       s=25, alpha=1.0, edgecolors='none')

    ax.set_xlabel(r'$\log_{10}(p - p_c)$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Order\ Parameters})$')
    ax.set_title(f'Critical Scaling: ER Positive ($c={c}, R={R}$)')

    ax.legend(frameon=False, loc='upper left')
    ax.grid(False)

    filename_img = f"er_pos_critical_indices_c{c}_R{R}"
    plt.tight_layout()
    plt.savefig(f'{filename_img}.png', dpi=300)
    plt.savefig(f'{filename_img}.pdf')
    print(f"\n绘图完成: {filename_img}.png")


if __name__ == "__main__":
    main()
