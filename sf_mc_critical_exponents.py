import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.optimize import root_scalar
import time


gammas = [2.5]
R = 3
k_c = 10000


def upper_inc_gamma(s, x):


    if s > 0:
        return sc.gammaincc(s, x) * sc.gamma(s)
    elif s == 0:
        return sc.exp1(x)
    else:

        return (upper_inc_gamma(s + 1.0, x) - (x ** s) * np.exp(-x)) / s


def T_tail(lambda_val, W, k_c):


    if W <= 0.0: return 0.0


    term1 = (k_c ** (1.0 - lambda_val)) / (lambda_val - 1.0) * (1.0 - np.exp(-W * k_c))

    s = 2.0 - lambda_val
    Gamma_val = upper_inc_gamma(s, W * k_c)
    term2 = (W ** (lambda_val - 1.0)) / (lambda_val - 1.0) * Gamma_val

    return term1 + term2


def get_F(W, gamma, k_c, Pk_disc, k_arr, C_norm):

    if W <= 0.0: return 0.0
    S_disc = np.sum(Pk_disc * (1.0 - (1.0 - W) ** k_arr))
    S_cont = C_norm * T_tail(gamma, W, k_c)
    return S_disc + S_cont


def get_G(W, gamma, k_c, Pk_disc, k_arr, C_norm, mean_k):

    if W <= 0.0: return 0.0
    S_disc = np.sum((k_arr / mean_k) * Pk_disc * (1.0 - (1.0 - W) ** k_arr))
    S_cont = (C_norm / mean_k) * T_tail(gamma - 1.0, W, k_c)
    return S_disc + S_cont


def solve_MP_hybrid_mcpl(p, gamma, k_c, Pk_disc, k_arr, C_norm, mean_k, R, max_iter=100000, tol=1e-13):
    W = np.full(R, 1e-5)
    for iteration in range(max_iter):
        W_new = np.zeros(R)
        sum_W = np.zeros(R + 1)
        for i in range(1, R + 1):
            sum_W[i] = min(sum_W[i - 1] + W[i - 1], 1.0)

        W_new[0] = p * get_G(sum_W[R], gamma, k_c, Pk_disc, k_arr, C_norm, mean_k)

        for r in range(1, R):
            G_prev = get_G(sum_W[r - 1], gamma, k_c, Pk_disc, k_arr, C_norm, mean_k)
            G_curr = get_G(sum_W[r], gamma, k_c, Pk_disc, k_arr, C_norm, mean_k)
            W_new[r] = (1.0 - p) * (G_curr - G_prev)

        if np.max(np.abs(W_new - W)) < tol:

            return W_new, sum_W, True
        W = W_new
    return W, sum_W, False


def exact_pc_equation(p, kappa_branching, R):
    term = (1.0 - p) * kappa_branching
    if abs(term - 1.0) < 1e-12:
        K = R
    else:
        K = (term ** R - 1.0) / (term - 1.0)
    return p * kappa_branching * K - 1.0


for gamma in gammas:
    print("=" * 70)
    print(f"🚀 启动万能连续积分引擎: gamma={gamma}, R={R}")

    k_arr = np.arange(1, k_c + 1)


    sum_Pk_raw = np.sum(k_arr ** (-gamma)) + (k_c ** (1.0 - gamma)) / (gamma - 1.0)
    C_norm = 1.0 / sum_Pk_raw
    Pk_disc = C_norm * (k_arr ** (-gamma))


    mean_k_disc = np.sum(Pk_disc * k_arr)
    mean_k_cont = C_norm * (k_c ** (2.0 - gamma)) / (gamma - 2.0)
    mean_k = mean_k_disc + mean_k_cont
    print(f"[*] 理论平均度 <k> = {mean_k:.6f}")

    pc = 0.0
    if gamma > 3.0:

        k2_disc = np.sum(Pk_disc * (k_arr ** 2))
        k2_cont = C_norm * (k_c ** (3.0 - gamma)) / (gamma - 3.0)
        k2_mean = k2_disc + k2_cont
        kappa_branching = k2_mean / mean_k
        print(f"[*] 分支因子 κ = {kappa_branching:.6f}")


        res = root_scalar(exact_pc_equation, args=(kappa_branching, R), bracket=[1e-6, 0.9999], method='brentq')
        pc = res.root
        print(f"[*] 理论渗流阈值 p_c = {pc:.8f}")
    else:
        print(f"[*] 极度反常区间，理论渗流阈值 p_c = 0")


    if pc == 0.0:
        dps = np.logspace(-10.0, -1.0, 100)
        ps = dps
    else:
        dps = np.logspace(-10.0, -1.0, 100)
        ps = pc + dps


    data = {'S+': [], 'S': [], 'U+': [], 'U': [], 'valid_dp': []}

    print("开始扫锚超临界区域...")
    t0 = time.time()
    for p, dp in zip(ps, dps):
        if p >= 1.0: continue

        W_converged, sum_W, is_conv = solve_MP_hybrid_mcpl(p, gamma, k_c, Pk_disc, k_arr, C_norm, mean_k, R)

        V_R = min(sum_W[R], 1.0)
        V_R_minus_1 = min(sum_W[R - 1], 1.0) if R > 1 else 0.0

        F_V_R = get_F(V_R, gamma, k_c, Pk_disc, k_arr, C_norm)
        F_V_R_minus_1 = get_F(V_R_minus_1, gamma, k_c, Pk_disc, k_arr, C_norm)


        S_plus = p * F_V_R
        S_scgc = (S_plus * S_plus) / p if p > 0 else 0

        U_plus = (1.0 - p) * F_V_R_minus_1
        U_scgc = (U_plus * U_plus) / (1.0 - p) if p < 1.0 else 0

        if S_plus > 1e-12:
            data['valid_dp'].append(dp)
            data['S+'].append(S_plus)
            data['S'].append(S_scgc)
            data['U+'].append(U_plus)
            data['U'].append(U_scgc)

    print(f"计算耗时: {time.time() - t0:.2f} 秒")


    valid_dp = np.array(data['valid_dp'])
    log_dp = np.log10(valid_dp)
    dp_mid = 10 ** ((log_dp[:-1] + log_dp[1:]) / 2)

    beta_eff = {}
    for key in ['S+', 'S', 'U+', 'U']:
        log_val = np.log10(np.array(data[key]))
        beta_eff[key] = np.diff(log_val) / np.diff(log_dp)


    if gamma <= 3.0:
        eta_R = (gamma - 2.0) ** (R - 1)
        t_S_plus = 1.0 + eta_R / (1.0 - (gamma - 2.0) ** R)
        t_S = 2.0 * t_S_plus - 1.0
        t_U_plus = ((gamma - 2.0) ** (R - 2)) / (1.0 - (gamma - 2.0) ** R)
        t_U = 2.0 * t_U_plus
    elif 3.0 < gamma <= 4.0:
        t_S_plus = t_U_plus = 1.0 / (gamma - 3.0)
        t_S = t_U = 2.0 / (gamma - 3.0)
    else:
        t_S_plus = t_U_plus = 1.0
        t_S = t_U = 2.0


    plt.figure(figsize=(10, 7))

    plt.plot(dp_mid, beta_eff['S+'], 'o-', color='blue', alpha=0.7, label=r'Num $\beta_{eff}$ $S^{(+)}$')
    plt.plot(dp_mid, beta_eff['S'], 's-', color='red', alpha=0.7, label=r'Num $\beta_{eff}$ $S$')
    plt.plot(dp_mid, beta_eff['U+'], '^-', color='green', alpha=0.7, label=r'Num $\beta_{eff}$ $U^{(+)}$')
    plt.plot(dp_mid, beta_eff['U'], 'd-', color='purple', alpha=0.7, label=r'Num $\beta_{eff}$ $U$')

    plt.axhline(t_S_plus, color='blue', ls='--', lw=2, label=f'Th $\\beta_{{S^+}} = {t_S_plus:.3f}$')
    plt.axhline(t_S, color='red', ls='--', lw=2, label=f'Th $\\beta_S = {t_S:.3f}$')
    plt.axhline(t_U_plus, color='green', ls='--', lw=2, label=f'Th $\\beta_{{U^+}} = {t_U_plus:.3f}$')
    plt.axhline(t_U, color='purple', ls='--', lw=2, label=f'Th $\\beta_U = {t_U:.3f}$')

    plt.xscale('log')
    all_effs = np.concatenate([beta_eff['S+'], beta_eff['S'], beta_eff['U+'], beta_eff['U']])
    all_ths = [t_S_plus, t_S, t_U_plus, t_U]
    y_min = max(0, min(np.min(all_effs) - 0.5, min(all_ths) - 0.5))
    y_max = max(np.max(all_effs) + 0.5, max(all_ths) + 0.5)
    plt.ylim(y_min, y_max)

    x_label = r'Connection Probability $p$' if pc == 0 else r'Distance to critical point $\Delta p = p - p_c$'
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(r'Effective Exponent $\beta_{eff}$', fontsize=14)
    plt.title(f'MCPL Full Asymptotic Exponents\n($\gamma={gamma}, R={R}$)', fontsize=15)
    plt.legend(fontsize=11, loc='upper right', ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    filename = f'mcpl_final_exponents_gamma_{gamma}_R_{R}.png'
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✅ 生成全解析图表: {filename}\n")

print("所有区间（极度反常、弱反常、平均场）的终极计算使命圆满完成！")
