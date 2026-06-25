import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numba import njit
from typing import TextIO


@njit
def min_accumulate ( Est_mu_arm):
    n = len(Est_mu_arm)
    running_min_arm = np.empty(n)
    
    running_min_arm[0] = Est_mu_arm[0]
    
    for i in range(1, n):
        if Est_mu_arm[i] < running_min_arm[i-1]:
            running_min_arm[i] = Est_mu_arm[i]
        else:
            running_min_arm[i] = running_min_arm[i-1]

    return running_min_arm
    

@njit #(forceobj=True)
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

        mask = Est_mu_arm1[i-1] >= Est_mu_arm2[i-1]

        if mask:
            cum1 += 1
            X_arm1[i] = X_arm1[i-1] + (mu1*interval + dB1[arm1_indx])
            arm1_indx += 1
            X_arm2[i] = X_arm2[i-1]
        else:
            cum2 += 1
            X_arm2[i] = X_arm2[i-1] + (mu2*interval + dB2[arm2_indx])
            arm2_indx += 1
            X_arm1[i] = X_arm1[i-1]

        E_arm1[i] = mask
        E_arm2[i] = 1 - mask

        Est_mu_arm1[i] = X_arm1[i] / (1 + cum1*interval)
        Est_mu_arm2[i] = X_arm2[i] / (1 + cum2*interval)

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

    return running_min_arm1, running_min_arm2, Est_mu_arm1, Est_mu_arm2, E_arm1, E_arm2, index1, index2



Tmax = 1000 #0.5 #0.2
mu1 = 0.5
mu2 = 1.2
interval = .1 #.001
alpha = -0.11 #-0.01 #-0.005

# Break down the time into small intervals
time = np.arange(0, Tmax, interval)

# This function is the main subroutine of Ankur's paper
# @njit
def ExchangeArms(mu1, mu2, time, r1, r2, alpha = 0.0):

    n = len(time)
    dt = time[5] - time[4]

    rng = np.random.default_rng(r1)
    # pre-generate Brownian increments
    dB1 = rng.normal(0, np.sqrt(interval), size=n)
    rng = np.random.default_rng(r2)
    dB2 = rng.normal(0, np.sqrt(interval), size=n)

    # print(np.mean(dB1), np.mean(dB2))
    # print(np.sum(dB1 > 0), np.sum(dB1 < 0))
    # print(np.sum(dB2 > 0), np.sum(dB2 < 0))
    
    
    res = FastExchangeArms(mu1, mu2, time, alpha, dB1, dB2, interval)
    
    running_min_arm1 = res[0]
    running_min_arm2 = res[1]
    Est_mu_arm1 = res[2]
    Est_mu_arm2 = res[3]
    E_arm1 = res[4]
    E_arm2 = res[5]
    index1 = res[6]
    index2 = res[7]
    
    # # print(running_min_arm1)
    # df = pd.DataFrame ({
    #                 "min_arm1":running_min_arm1,
    #                 "min_arm2":running_min_arm2,
    #                 "mu_1": Est_mu_arm1,
    #                 "mu_2": Est_mu_arm2,
    #                 "effort_arm1":np.cumsum(E_arm1),
    #                 "effort_arm2":np.cumsum(E_arm2),
    #                 "BM1": np.cumsum(dB1),
    #                 "BM2": np.cumsum(dB2)
    #                 })
    # # # df.to_excel("arm.xlsx", index=False)
    
    
    # fig = plt.figure(figsize=(10,6))
    # fig.canvas.manager.set_window_title("Pre Interaction")
    # # Plot estimated means (or running processes)
    # plt.plot(df["mu_1"], label=f"Arm 1 = {mu1}", color="blue")
    # plt.plot(df["mu_2"], label=f"Arm 2 = {mu2}", color="green")

    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title("Arm Estimates / Brownian Motion with Drift")

    # plt.legend()
    # plt.grid(True)

    # plt.show()

        
    # print(f"Last element of mu arm1 = {Est_mu_arm1[-1]}, Last element of mu arm2 = {Est_mu_arm2[-1]}")
    return running_min_arm1, running_min_arm2, Est_mu_arm1, Est_mu_arm2, E_arm1, E_arm2,  index1, index2 
    # return X_arm1, X_arm2

#This function is the part which simulates the dynamics post interaction.
@njit 
def FastUpdateReceiver(
        recv_arm1,     
        recv_arm2,
        jmp_indx ,
        recv_Jmp_E_arm1,
        recv_Jmp_E_arm2,
        prov_Jmp_E_arm1,
        prov_Jmp_E_arm2,
        level,
        dB1,dB2, 
        mu1, mu2, 
        time = time
):
    n = len(time)
    dt = time[1] - time[0]

    E_arm1 = np.zeros(n)
    E_arm2 = np.zeros(n)

    running_min_arm1 = recv_arm1 # This will have the running minimum after the shock
    running_min_arm2 = recv_arm2 # This will have the running minimum after the shock
    
    print(f"recv_Jmp_E_arm1 ,  prov_Jmp_E_arm1 = {recv_Jmp_E_arm1,  prov_Jmp_E_arm1} \n")
    print(f"recv_Jmp_E_arm2 ,  prov_Jmp_E_arm2 = {recv_Jmp_E_arm2,  prov_Jmp_E_arm2} \n")
    E_arm1[jmp_indx] = recv_Jmp_E_arm1 + prov_Jmp_E_arm1
    E_arm2[jmp_indx] = recv_Jmp_E_arm2 + prov_Jmp_E_arm2
    
    recv_arm1[jmp_indx] = (
        recv_arm1[jmp_indx] * (recv_Jmp_E_arm1*dt + 1) + level*(prov_Jmp_E_arm1*dt +1)
        )/(E_arm1[jmp_indx] *dt + 1)

    recv_arm2[jmp_indx] = (
        recv_arm2[jmp_indx] * (recv_Jmp_E_arm2 *dt + 1) + level*(prov_Jmp_E_arm2*dt +1)
        )/(E_arm2[jmp_indx] *dt + 1)
    
    
    cumE1 = np.cumsum(E_arm1)[jmp_indx]
    cumE2 = np.cumsum(E_arm2)[jmp_indx]
    
    for i in range(jmp_indx+1, n) :
        mask = recv_arm1[i-1] >= recv_arm2[i-1]
        
        if mask:
            cumE1 += 1
        else:
            cumE2 += 1

        E_arm1[i] = mask
        E_arm2[i] = 1-mask

        recv_arm1[i] = recv_arm1[i-1]*(1 + (cumE1-1)*dt) + mask * (mu1*dt + dB1[i]) # BM receieved upto this time for arm 1
        E_arm1[i] = mask
        recv_arm2[i] = recv_arm2[i-1]*(1 + (cumE2 -1)*dt) + (1-mask) * (mu2*dt + dB2[i]) # BM receieved upto this time for arm 2
        E_arm2[i] = (1-mask)
        
        recv_arm1[i] = recv_arm1[i]/(1+cumE1*dt)     # mu_1
        recv_arm2[i] = recv_arm2[i]/(1+cumE2*dt)     # mu_2

    
    return recv_arm1, recv_arm2

def UpdateReceiver(
        recv_arm1,     
        recv_arm2,
        jmp_indx ,
        recv_Jmp_E_arm1,
        recv_Jmp_E_arm2,
        prov_Jmp_E_arm1,
        prov_Jmp_E_arm2,
        level,
        r1,r2,
        mu1, mu2,  
        time = time
):

    n = len(time)
    dt = time[1] - time[0]

    rng = np.random.default_rng(r1)
    # pre-generate Brownian increments
    dB1 = rng.normal(0, np.sqrt(dt), size=n)
    rng = np.random.default_rng(r2)
    dB2 = rng.normal(0, np.sqrt(dt), size=n)


    recv_arm1, recv_arm2 = FastUpdateReceiver(
        recv_arm1,     
        recv_arm2,
        jmp_indx ,
        recv_Jmp_E_arm1,
        recv_Jmp_E_arm2,
        prov_Jmp_E_arm1,
        prov_Jmp_E_arm2,
        level,
        dB1,dB2,
        mu1, mu2,  
        time
    )
    
    running_min_arm1 = np.minimum.accumulate(recv_arm1)
    running_min_arm2 = np.minimum.accumulate(recv_arm2)

    # # pre-generate Brownian increments
    # dB1 = np.random.normal(0, np.sqrt(dt), size=n)
    # dB2 = np.random.normal(0, np.sqrt(dt), size=n)
    
    # E_arm1 = np.zeros(n)
    # E_arm2 = np.zeros(n)

    # running_min_arm1 = recv_arm1 # This will have the running minimum after the shock
    # running_min_arm2 = recv_arm2 # This will have the running minimum after the shock
    
    # print(f"recv_Jmp_E_arm1 ,  prov_Jmp_E_arm1 = {recv_Jmp_E_arm1,  prov_Jmp_E_arm1} \n")
    # print(f"recv_Jmp_E_arm2 ,  prov_Jmp_E_arm2 = {recv_Jmp_E_arm2,  prov_Jmp_E_arm2} \n")
    # E_arm1[jmp_indx] = recv_Jmp_E_arm1 + prov_Jmp_E_arm1
    # E_arm2[jmp_indx] = recv_Jmp_E_arm2 + prov_Jmp_E_arm2
    
    # recv_arm1[jmp_indx] = (
    #     recv_arm1[jmp_indx] * (recv_Jmp_E_arm1*dt + 1) + level*(prov_Jmp_E_arm1*dt +1)
    #     )/(E_arm1[jmp_indx] *dt + 1)

    # recv_arm2[jmp_indx] = (
    #     recv_arm2[jmp_indx] * (recv_Jmp_E_arm2 *dt + 1) + level*(prov_Jmp_E_arm2*dt +1)
    #     )/(E_arm2[jmp_indx] *dt + 1)
    
    
    # cumE1 = np.cumsum(E_arm1)[jmp_indx]
    # cumE2 = np.cumsum(E_arm2)[jmp_indx]
    
    # for i in range(jmp_indx+1, n) :
    #     mask = recv_arm1[i-1] >= recv_arm2[i-1]
        
    #     if mask:
    #         cumE1 += 1
    #     else:
    #         cumE2 += 1

    #     E_arm1[i] = mask
    #     E_arm2[i] = 1-mask

    #     recv_arm1[i] = recv_arm1[i-1]*(1 + (cumE1-1)*dt) + mask * (mu1*dt + dB1[i]) # BM receieved upto this time for arm 1
    #     E_arm1[i] = mask
    #     recv_arm2[i] = recv_arm2[i-1]*(1 + (cumE2 -1)*dt) + (1-mask) * (mu2*dt + dB2[i]) # BM receieved upto this time for arm 2
    #     E_arm2[i] = (1-mask)
        
    #     recv_arm1[i] = recv_arm1[i]/(1+cumE1*dt)     # mu_1
    #     recv_arm2[i] = recv_arm2[i]/(1+cumE2*dt)     # mu_2

    # running_min_arm1 = np.minimum.accumulate(recv_arm1)
    # running_min_arm2 = np.minimum.accumulate(recv_arm2)


    # df = pd.DataFrame ({
    #                 "min_arm1":running_min_arm1,
    #                 "min_arm2":running_min_arm2,
    #                 "mu_1": recv_arm1,
    #                 "mu_2": recv_arm2,
    #                 "effort_arm1":np.cumsum(E_arm1),
    #                 "effort_arm2":np.cumsum(E_arm2),
    #                 "BM1": np.cumsum(dB1),
    #                 "BM2": np.cumsum(dB2)
    #                 })

    # fig = plt.figure(figsize=(10,6))
    # fig.canvas.manager.set_window_title("Post Interaction")
    # # Plot estimated means (or running processes)
    # plt.plot(df["mu_1"], label=f"Arm 1={mu1}", color="blue")
    # plt.plot(df["mu_2"], label=f"Arm 2={mu2}", color="green")

    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title("Arm Estimates / Brownian Motion with Drift")

    # plt.legend()
    # plt.grid(True)

    
    # plt.show()
    
    return recv_arm1, recv_arm2



#Call before the interaction
# running_min_arm1_agnt1, running_min_arm2_agnt1, E_arm1_a1, E_arm2_a1, indx1_a1, indx2_a1 = ExchangeArms(mu1,mu2, time, alpha)

def Communication(mu1:float, mu2:float, alpha:float, outfile:TextIO):
    if os.path.exists("symmetric.csv"):
        os.remove("symmetric.csv")

    if os.path.exists("symmetric_negJump.csv"):
        os.remove("symmetric_negJump.csv")
            
    ss = np.random.SeedSequence(1230987654)


    for child in ss.spawn(50000):
        r1, r2, r3, r4 = child.generate_state(4)

        (running_min_arm1_agnt1, running_min_arm2_agnt1, 
        Est_mu_arm1_a1, Est_mu_arm2_a1,  
        E_arm1_a1, E_arm2_a1, 
        indx1_a1, indx2_a1) = ExchangeArms(mu1,mu2, time, r1, r2, alpha)
        
        (running_min_arm1_agnt2, running_min_arm2_agnt2,
        Est_mu_arm1_a2, Est_mu_arm2_a2, 
        E_arm1_a2, E_arm2_a2, 
        indx1_a2, indx2_a2) = ExchangeArms(mu1,mu2, time, r3, r4, alpha)

        #Calculate the jump index

        print(f"indx1_a1={indx1_a1}, indx2_a1={indx2_a1}, indx1_a2={indx1_a2}, indx2_a2={indx2_a2}")
        print(f"E_arm1_a1, E_arm2_a1={E_arm1_a1, E_arm2_a1}, E_arm1_a2, E_arm2_a2={E_arm1_a2, E_arm2_a2} ")
        max_a1 = max(indx1_a1, indx2_a1)
        max_a2 = max(indx1_a2, indx2_a2)

        giver = 2 if max_a1 > max_a2 else 1

        if (giver == 2) :
            jump_idx = max_a2
        else:
            jump_idx = max_a1

        print(f"giver={giver}, jump_idx={jump_idx}")

        # print (E_arm1_a2)

        if (jump_idx < 0):
            with open("symmetric_negJump.csv", "a") as f:
                val1 = 1 if Est_mu_arm1_a1[-1] < Est_mu_arm2_a1[-1] else 0
                val2 = 1 if Est_mu_arm1_a2[-1] < Est_mu_arm2_a2[-1] else 0
                
                f.write(
                    f"{Est_mu_arm1_a1[-1]}, {Est_mu_arm2_a1[-1]}, "
                    f"{val1}, "
                    f"{Est_mu_arm1_a2[-1]}, {Est_mu_arm2_a2[-1]}, "
                    f"{val2} \n "
                )
            continue

                
        receiver_min_arm1_agnt1, receiver_min_arm2_agnt1 = UpdateReceiver(
            running_min_arm1_agnt1,
            running_min_arm2_agnt1,
            jump_idx,
            np.cumsum(E_arm1_a1)[jump_idx],
            np.cumsum(E_arm2_a1)[jump_idx],
            np.cumsum(E_arm1_a2)[jump_idx],
            np.cumsum(E_arm2_a2)[jump_idx],
            alpha,
            r1,r2,
            mu1,mu2
        )
        
        receiver_min_arm1_agnt2, receiver_min_arm2_agnt2 = UpdateReceiver(
            running_min_arm1_agnt2,
            running_min_arm2_agnt2,
            jump_idx,
            np.cumsum(E_arm1_a2)[jump_idx],
            np.cumsum(E_arm2_a2)[jump_idx],
            np.cumsum(E_arm1_a1)[jump_idx],
            np.cumsum(E_arm2_a1)[jump_idx],
            alpha,
            r3,r4,
            mu1, mu2
        )
        
        with open("symmetric.csv", "a") as f:
            val1 = 1 if receiver_min_arm1_agnt1[-1] < receiver_min_arm2_agnt1[-1] else 0
            val2 = 1 if receiver_min_arm1_agnt2[-1] < receiver_min_arm2_agnt2[-1] else 0

            valb1 = 1 if Est_mu_arm1_a1[jump_idx] < Est_mu_arm2_a1[jump_idx] else 0
            valb2 = 1 if Est_mu_arm1_a2[jump_idx] < Est_mu_arm2_a2[jump_idx] else 0
            
            f.write(
                f"{receiver_min_arm1_agnt1[-1]}, {receiver_min_arm2_agnt1[-1]}, "
                f"{val1}, "
                f"{receiver_min_arm1_agnt2[-1]}, {receiver_min_arm2_agnt2[-1]}, "
                f"{val2}, "
                f"{giver}, "
                f"{np.cumsum(E_arm1_a2)[jump_idx] + np.cumsum(E_arm2_a2)[jump_idx]}, "
                f"{np.cumsum(E_arm1_a1)[jump_idx] + np.cumsum(E_arm2_a1)[jump_idx]}, "
                f"{jump_idx} ,"
                f"{valb1|valb2}, {val1|val2}, {1 - ((valb1|valb2)^(val1|val2))} \n"
            )

    if os.path.exists("symmetric.csv"):
        ff_symmetric = pd.read_csv("symmetric.csv")
        ff_negjumps = pd.read_csv("symmetric_negJump.csv")
            
        count1 = ((ff_symmetric.iloc[:, 2] == 0) & 
                (ff_symmetric.iloc[:, 5] == 0)).sum()
        
        
        count2 = ((ff_negjumps.iloc[:, 2] == 0) & 
                (ff_negjumps.iloc[:, 5] == 0)).sum()
        
        outfile.write(fr"{alpha} & {ff_symmetric.shape[0]} & {count1} & {count2 + count1} \\ \hline")
        outfile.write("\n")
        
        print(f"Count = {count1}, Rows = {ff_symmetric.shape[0]} \n")
        print(f"Count2 = {count2+ count1} \n")

    else :
        ff_negjumps = pd.read_csv("symmetric_negJump.csv")
        
        count1 = 0
        count2 = ((ff_negjumps.iloc[:, 2] == 0) & 
                (ff_negjumps.iloc[:, 5] == 0)).sum()
        
        outfile.write(fr"{alpha} & {count1} & {count1} & {count2 + count1} \\ \hline")
        outfile.write("\n")
        
        print(f"Count = \n", 0)
        print(f"Count2 = {count2} \n")
        
# main block 

fileName = "Symmetric_Result_"+ str(mu1)+"_"+ str(mu2)+".txt"

if os.path.exists(fileName):
    os.remove(fileName) 
       
    
ll = [-3.0, -0.46, -0.41, -0.36, -0.31, -0.26, -0.21, -0.16, -0.11, -0.06, -0.03, -0.01]
#ll = [-3.0, -0.46, -0.41, -0.36]

file: TextIO = open(fileName, "a")
file.write(r"""
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
""")

for ele in ll:
    alpha = ele     
    Communication(mu1, mu2, alpha, file)
    
file.write(r"""
\end{tabular}
}
}
\end{table}
""")

file.close()
