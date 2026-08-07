import concurrent.futures
import matplotlib
import numpy as np
import os
import pandas as pd
from numba import njit
from typing import TextIO

matplotlib.use("TkAgg")

# --- Numba Compiled Subroutines (Ultra-Fast Math Execution) ---
@njit
def min_accumulate(Est_mu_arm):
    n = len(Est_mu_arm)
    running_min_arm = np.empty(n)
    running_min_arm[0] = Est_mu_arm[0]
    for i in range(1, n):
        if Est_mu_arm[i] < running_min_arm[i - 1]:
            running_min_arm[i] = Est_mu_arm[i]
        else:
            running_min_arm[i] = running_min_arm[i - 1]
    return running_min_arm


@njit
def FastExchangeArms(mu1, mu2, time, alpha, dB1, dB2, interval):
    n = len(time)
    X_arm1 = np.zeros(n)
    X_arm2 = np.zeros(n)
    E_arm1 = np.zeros(n)
    E_arm2 = np.zeros(n)
    Est_mu_arm1 = np.zeros(n)
    Est_mu_arm2 = np.zeros(n)

    arm1_indx = 0
    arm2_indx = 0
    cum1 = 0
    cum2 = 0

    for i in range(1, n):
        mask = Est_mu_arm1[i - 1] >= Est_mu_arm2[i - 1]
        if mask:
            cum1 += 1
            X_arm1[i] = X_arm1[i - 1] + (mu1 * interval + dB1[arm1_indx])
            arm1_indx += 1
            X_arm2[i] = X_arm2[i - 1]
        else:
            cum2 += 1
            X_arm2[i] = X_arm2[i - 1] + (mu2 * interval + dB2[arm2_indx])
            arm2_indx += 1
            X_arm1[i] = X_arm1[i - 1]

        E_arm1[i] = mask
        E_arm2[i] = 1 - mask
        Est_mu_arm1[i] = X_arm1[i] / (1 + cum1 * interval)
        Est_mu_arm2[i] = X_arm2[i] / (1 + cum2 * interval)

    running_min_arm1 = min_accumulate(Est_mu_arm1)
    running_min_arm2 = min_accumulate(Est_mu_arm2)

    index1 = -1
    for i in range(n):
        if running_min_arm1[i] <= alpha:
            index1 = i
            break

    index2 = -1
    for i in range(n):
        if running_min_arm2[i] <= alpha:
            index2 = i
            break

    return (
        running_min_arm1,
        running_min_arm2,
        Est_mu_arm1,
        Est_mu_arm2,
        E_arm1,
        E_arm2,
        index1,
        index2,
    )


@njit
def FastUpdateReceiver(
    recv_arm1,
    recv_arm2,
    jmp_indx,
    recv_Jmp_E_arm1,
    recv_Jmp_E_arm2,
    prov_Jmp_E_arm1,
    prov_Jmp_E_arm2,
    level,
    dB1,
    dB2,
    mu1,
    mu2,
    dt,
):
    n = len(recv_arm1)
    E_arm1 = np.zeros(n)
    E_arm2 = np.zeros(n)

    E_arm1[jmp_indx] = recv_Jmp_E_arm1 + prov_Jmp_E_arm1
    E_arm2[jmp_indx] = recv_Jmp_E_arm2 + prov_Jmp_E_arm2

    recv_arm1[jmp_indx] = (
        recv_arm1[jmp_indx] * (recv_Jmp_E_arm1 * dt + 1)
        + level * (prov_Jmp_E_arm1 * dt + 1)
    ) / (E_arm1[jmp_indx] * dt + 1)
    recv_arm2[jmp_indx] = (
        recv_arm2[jmp_indx] * (recv_Jmp_E_arm2 * dt + 1)
        + level * (prov_Jmp_E_arm2 * dt + 1)
    ) / (E_arm2[jmp_indx] * dt + 1)

    cumE1 = recv_Jmp_E_arm1 + prov_Jmp_E_arm1
    cumE2 = recv_Jmp_E_arm2 + prov_Jmp_E_arm2

    for i in range(jmp_indx + 1, n):
        mask = recv_arm1[i - 1] >= recv_arm2[i - 1]
        if mask:
            cumE1 += 1
        else:
            cumE2 += 1

        E_arm1[i] = mask
        E_arm2[i] = 1 - mask

        recv_arm1[i] = (
            recv_arm1[i - 1] * (1 + (cumE1 - 1) * dt) + mask * (mu1 * dt + dB1[i])
        ) / (1 + cumE1 * dt)
        recv_arm2[i] = (
            recv_arm2[i - 1] * (1 + (cumE2 - 1) * dt)
            + (1 - mask) * (mu2 * dt + dB2[i])
        ) / (1 + cumE2 * dt)

    return recv_arm1, recv_arm2


# --- Global Configurations ---
Tmax = 1000
mu1 = 0.1
mu2 = 1.2
interval = 0.001
time = np.arange(0, Tmax, interval)


def ExchangeArms(mu1, mu2, time, r1, r2, alpha=0.0):
    n = len(time)
    rng = np.random.default_rng(r1)
    dB1 = rng.normal(0, np.sqrt(interval), size=n)
    rng = np.random.default_rng(r2)
    dB2 = rng.normal(0, np.sqrt(interval), size=n)
    return FastExchangeArms(mu1, mu2, time, alpha, dB1, dB2, interval)


def UpdateReceiver(
    recv_arm1,
    recv_arm2,
    jmp_indx,
    recv_Jmp_E_arm1,
    recv_Jmp_E_arm2,
    prov_Jmp_E_arm1,
    prov_Jmp_E_arm2,
    level,
    r1,
    r2,
):
    n = len(time)
    dt = interval
    rng = np.random.default_rng(r1)
    dB1 = rng.normal(0, np.sqrt(dt), size=n)
    rng = np.random.default_rng(r2)
    dB2 = rng.normal(0, np.sqrt(dt), size=n)
    return FastUpdateReceiver(
        recv_arm1,
        recv_arm2,
        jmp_indx,
        recv_Jmp_E_arm1,
        recv_Jmp_E_arm2,
        prov_Jmp_E_arm1,
        prov_Jmp_E_arm2,
        level,
        dB1,
        dB2,
        mu1,
        mu2,
        dt,
    )


# --- Symmetric Worker Engine ---
def run_single_simulation(seeds, mu1, mu2, alpha):
    """Handles a single symmetric dual-update step completely in system memory."""
    r1, r2, r3, r4 = seeds

    # Execute initial arm allocations
    res_a1 = ExchangeArms(mu1, mu2, time, r1, r2, alpha)
    res_a2 = ExchangeArms(mu1, mu2, time, r3, r4, alpha)

    indx1_a1, indx2_a1 = res_a1[6], res_a1[7]
    indx1_a2, indx2_a2 = res_a2[6], res_a2[7]

    max_a1 = max(indx1_a1, indx2_a1)
    max_a2 = max(indx1_a2, indx2_a2)

    giver = 2 if max_a1 > max_a2 else 1
    jump_idx = max_a2 if giver == 2 else max_a1

    # Non-communicating fallback path
    if jump_idx < 0:
        val1 = 1 if res_a1[2][-1] < res_a1[3][-1] else 0
        val2 = 1 if res_a2[2][-1] < res_a2[3][-1] else 0
        return (
            "neg_jump",
            (res_a1[2][-1], res_a1[3][-1], val1, res_a2[2][-1], res_a2[3][-1], val2),
        )

    # Symmetric updating path: both receivers update their estimates mutually
    recv_min1_a1, recv_min2_a1 = UpdateReceiver(
        res_a1[0].copy(),
        res_a1[1].copy(),
        jump_idx,
        np.cumsum(res_a1[4])[jump_idx],
        np.cumsum(res_a1[5])[jump_idx],
        np.cumsum(res_a2[4])[jump_idx],
        np.cumsum(res_a2[5])[jump_idx],
        alpha,
        r1,
        r2,
    )

    recv_min1_a2, recv_min2_a2 = UpdateReceiver(
        res_a2[0].copy(),
        res_a2[1].copy(),
        jump_idx,
        np.cumsum(res_a2[4])[jump_idx],
        np.cumsum(res_a2[5])[jump_idx],
        np.cumsum(res_a1[4])[jump_idx],
        np.cumsum(res_a1[5])[jump_idx],
        alpha,
        r3,
        r4,
    )

    val1 = 1 if recv_min1_a1[-1] < recv_min2_a1[-1] else 0
    val2 = 1 if recv_min1_a2[-1] < recv_min2_a2[-1] else 0

    valb1 = 1 if res_a1[2][jump_idx] < res_a1[3][jump_idx] else 0
    valb2 = 1 if res_a2[2][jump_idx] < res_a2[3][jump_idx] else 0

    record = (
        recv_min1_a1[-1],
        recv_min2_a1[-1],
        val1,
        recv_min1_a2[-1],
        recv_min2_a2[-1],
        val2,
        giver,
        np.cumsum(res_a2[4])[jump_idx] + np.cumsum(res_a2[5])[jump_idx],
        np.cumsum(res_a1[4])[jump_idx] + np.cumsum(res_a1[5])[jump_idx],
        jump_idx,
        (valb1 | valb2),
        (val1 | val2),
        (1 - ((valb1 | valb2) ^ (val1 | val2))),
    )

    return "symmetric", record


def Communication(mu1: float, mu2: float, alpha: float, outfile: TextIO):
    print(f"Starting Multi-Core Symmetric Simulation for Alpha: {alpha}...")
    ss = np.random.SeedSequence(1230987654)
    tasks = [child.generate_state(4) for child in ss.spawn(50000)]

    symmetric_records = []
    neg_jump_records = []

    # Parallel core pipeline execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_simulation, task, mu1, mu2, alpha)
            for task in tasks
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            dtype, record = future.result()
            if dtype == "symmetric":
                symmetric_records.append(record)
            else:
                neg_jump_records.append(record)

            if idx % 10000 == 0 and idx > 0:
                print(f"Completed {idx}/50000 jobs...")

    # Safe unified filesystem sweeps
    df_sym = pd.DataFrame(symmetric_records)
    df_neg = pd.DataFrame(neg_jump_records)

    df_sym.to_csv("symmetric.csv", index=False)
    df_neg.to_csv("symmetric_negJump.csv", index=False)

    count1 = (
        ((df_sym.iloc[:, 2] == 0) & (df_sym.iloc[:, 5] == 0)).sum()
        if not df_sym.empty
        else 0
    )
    count2 = (
        ((df_neg.iloc[:, 2] == 0) & (df_neg.iloc[:, 5] == 0)).sum()
        if not df_neg.empty
        else 0
    )

    outfile.write(
        fr"{alpha} & {df_sym.shape[0]} & {count1} & {count2 + count1} \\ \hline"
    )
    outfile.write("\n")
    print(f"Finished Alpha {alpha}. Total Rows written: {df_sym.shape[0]}")


# --- Runtime System Protection entry-point ---
if __name__ == "__main__":
    fileName = "Symmetric_Result_" + str(mu1) + "_" + str(mu2) + ".txt"
    if os.path.exists(fileName):
        os.remove(fileName)

    # ll = [
    #     -0.01,
    #     -0.001,
    #     -0.0001,
    #     -0.00001,
    #     -0.000001,
    #     -0.0000001,
    #     -0.00000001,
    #     -0.000000001,
    #     -0.0000000001,
    #     -0.00000000001,
    # ]

    ll = [
        -3.0,
        -0.46,
        -0.41,
        -0.36,
        -0.31,
        -0.26,
        -0.21,
        -0.16,
        -0.11,
        -0.06,
        -0.03,
        -0.01,
        -0.001,
        -0.0001,
        -0.00001,
        -0.000001,
        -0.0000001,
        -0.00000001,
        -0.000000001,
        -0.0000000001,
        -0.00000000001
    ]


    with open(fileName, "a") as file:
        file.write(
            r"""
\begin{table}[h!]
\centering
{\tiny
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|p{1cm}|p{1.1cm}|p{1.1cm}|}
\hline
$\alpha$
& {\fontsize{7}{9}\selectfont Total Communication}
& {\fontsize{7}{9}\selectfont Symmetric sub optimal}
& {\fontsize{7}{9}\selectfont Symmetric + Non Communicating}
\\
\hline
"""
        )

        for ele in ll:
            Communication(mu1, mu2, ele, file)

        file.write(
            r"""
\end{tabular}
}
}
\end{table}
"""
        )