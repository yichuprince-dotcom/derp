import numpy as np
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.special import gammaincc
from scipy.special import gamma as gamma_func
import time


output_dir = "results_gamma_scan"
os.makedirs(output_dir, exist_ok=True)


gammas_theory = np.round(np.arange(2.2, 3.51, 0.01), 2)
gammas_sim = np.round(np.arange(2.2, 3.51, 0.1), 2)

R = 2
k_c = 10000


def get_F(W, gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3):
    if W <= 0.0: return 0.0
    term_W = 1.0 - W
    S_disc = np.sum(Pk_disc * (1.0 - term_W ** k_arr))

    x = W * k_c
    G3 = gammaincc(s3, x) * gamma_s3
    term1 = (k_c ** (1.0 - gamma)) * (1.0 - np.exp(-x)) / (gamma - 1.0)
    term2 = W * (k_c ** (2.0 - gamma)) * np.exp(-x) / ((gamma - 1.0) * (gamma - 2.0))
    term3 = (W ** (gamma - 1.0)) * G3 / ((gamma - 1.0) * (gamma - 2.0))

    S_cont = C_norm * (term1 + term2 - term3)
    return S_disc + S_cont


def solve_MP_hybrid(p, gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3, R, max_iter=80000, tol=1e-13):
    W = np.full(R, 1e-5)
    for iteration in range(max_iter):
        W_new = np.zeros(R)
        sum_W = np.zeros(R + 1)
        for i in range(1, R + 1):
            sum_W[i] = min(sum_W[i - 1] + W[i - 1], 1.0)

        F_W_R = get_F(sum_W[R], gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3)
        W_new[0] = p * F_W_R

        for r in range(1, R):
            F_prev = get_F(sum_W[r - 1], gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3)
            F_curr = get_F(sum_W[r], gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3)
            W_new[r] = (1.0 - p) * (F_curr - F_prev)

        if np.max(np.abs(W_new - W)) < tol:
            return W_new, True
        W = W_new
    return W, False


def exact_pc_equation(p, kappa, R):
    term = (1.0 - p) * kappa
    if abs(term - 1.0) < 1e-12:
        K = R
    else:
        K = (term ** R - 1.0) / (term - 1.0)
    return p * kappa * K - 1.0


sim_results = {}

print("�� 开始进行 Gamma 扫描与局部有效指数提取 (包含均场区域)...")

for gamma in gammas_sim:
    print("-" * 50)
    print(f"�� 当前仿真进度: gamma={gamma:.2f}")


    calc_gamma = gamma if abs(gamma - 3.0) > 1e-5 else gamma + 1e-7

    k_arr = np.arange(1, k_c + 1)
    s3 = 3.0 - calc_gamma
    gamma_s3 = gamma_func(s3)

    sum_Pk_raw = np.sum(k_arr ** (-calc_gamma)) + (k_c ** (1.0 - calc_gamma)) / (calc_gamma - 1.0)
    C_norm = 1.0 / sum_Pk_raw
    Pk_disc = C_norm * (k_arr ** (-calc_gamma))

    kappa_disc = np.sum(Pk_disc * k_arr)
    kappa_cont = C_norm * (k_c ** (2.0 - calc_gamma)) / (calc_gamma - 2.0)
    kappa = kappa_disc + kappa_cont

    res = root_scalar(exact_pc_equation, args=(kappa, R), bracket=[0.01, 0.99], method='brentq')
    pc = res.root

    delta_ps = np.logspace(-5.0, -1.0, 60)
    S_plus_list, S_scgc_list, valid_dp = [], [], []

    for dp in delta_ps:
        p = pc + dp
        if p >= 1.0: continue

        W_converged, _ = solve_MP_hybrid(p, calc_gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3, R)
        sum_W_R = min(np.sum(W_converged), 1.0)
        term_S = get_F(sum_W_R, calc_gamma, k_c, Pk_disc, k_arr, C_norm, s3, gamma_s3)
        S_plus = p * term_S

        if S_plus > 1e-12:
            valid_dp.append(dp)
            S_plus_list.append(S_plus)
            S_scgc_list.append(p * term_S * term_S)

    valid_dp = np.array(valid_dp)
    log_dp = np.log10(valid_dp)
    log_S_plus = np.log10(np.array(S_plus_list))
    log_S_scgc = np.log10(np.array(S_scgc_list))

    beta_eff_plus = np.diff(log_S_plus) / np.diff(log_dp)
    beta_eff_scgc = np.diff(log_S_scgc) / np.diff(log_dp)

    max_b_plus = np.max(beta_eff_plus) if len(beta_eff_plus) > 0 else np.nan
    max_b_scgc = np.max(beta_eff_scgc) if len(beta_eff_scgc) > 0 else np.nan

    sim_results[gamma] = (max_b_plus, max_b_scgc)
    print(f"   找到最高有效指数: Beta_S+ = {max_b_plus:.4f}, Beta_S = {max_b_scgc:.4f}")


print("\n�� 正在将数据写入文本文件...")
txt_path = os.path.join(output_dir, "gamma_exponent_results.txt")

theory_beta_plus_arr = []
theory_beta_scgc_arr = []

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("gamma\t理论_S+\t仿真_S+\t理论_S\t仿真_S\n")
    for g in gammas_theory:

        if g <= 3.0:
            th_plus = 1.0 / (g - 2.0)
            th_scgc = 2.0 / (g - 2.0)
        else:
            th_plus = 1.0
            th_scgc = 2.0

        theory_beta_plus_arr.append(th_plus)
        theory_beta_scgc_arr.append(th_scgc)


        idx = np.where(np.isclose(gammas_sim, g, atol=1e-4))[0]
        if len(idx) > 0:
            sim_g = gammas_sim[idx[0]]
            sm_plus, sm_scgc = sim_results[sim_g]
            sm_plus_str = f"{sm_plus:.4f}"
            sm_scgc_str = f"{sm_scgc:.4f}"
        else:
            sm_plus_str = "nan"
            sm_scgc_str = "nan"

        f.write(f"{g:.2f}\t{th_plus:.4f}\t{sm_plus_str}\t{th_scgc:.4f}\t{sm_scgc_str}\n")


print("�� 正在渲染对比图片...")

sim_max_beta_plus = [sim_results[g][0] for g in gammas_sim]
sim_max_beta_scgc = [sim_results[g][1] for g in gammas_sim]

plt.figure(figsize=(10, 6))


plt.plot(gammas_theory, theory_beta_plus_arr, 'b-', linewidth=2, label=r'Theory $\beta_{S^{(+)}}$')
plt.plot(gammas_theory, theory_beta_scgc_arr, 'r-', linewidth=2, label=r'Theory $\beta_{S}$')


plt.plot(gammas_sim, sim_max_beta_plus, 'bo', markersize=8, alpha=0.8, label=r'Simulation Max $\beta_{S^{(+)}}$')
plt.plot(gammas_sim, sim_max_beta_scgc, 'rs', markersize=8, alpha=0.8, label=r'Simulation Max $\beta_{S}$')


plt.axvline(x=3.0, color='gray', linestyle=':', linewidth=1.5, label='Mean-Field Boundary ($\gamma=3.0$)')

plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel(r'Effective Exponent $\beta_{max}$', fontsize=14)
plt.title(r'Effective Exponents ($\beta$) vs $\gamma$ (With Mean-Field Regime)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

img_path = os.path.join(output_dir, "gamma_vs_exponents_extended.png")
plt.savefig(img_path, dpi=300)
plt.close()

print(f"✅ 所有任务运行完毕！带 nan 填充的对齐结果已保存在 '{output_dir}/gamma_exponent_results.txt' 中。")
