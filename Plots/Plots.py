import matplotlib.pyplot as plt

DOF = [17062806, 34028006, 51138918, 68211558, 85136918, 102278262]
Ranks = [1, 2, 3, 4, 5, 6]

# All times are in seconds
Times1 = [5.36227, 4.60392, 4.21676, 4.50057, 4.56992, 4.77939] # Corresponds to DOF[0]
Times2 = [11.2400, 9.47284, 8.49174, 8.68912, 8.36811, 8.54736] # Corresponds to DOF[1]
Times3 = [15.4770, 13.4667, 11.7430, 12.1529, 11.6964, 11.6211] # Corresponds to DOF[2]
Times4 = [20.9097, 19.1745, 15.4112, 15.1178, 15.4682, 15.0777] # Corresponds to DOF[3]
Times5 = [24.7205, 20.3123, 17.8193, 18.6405, 17.3797, 18.2512] # Corresponds to DOF[4]
Times6 = [30.8935, 25.7645, 22.8770, 21.9028, 21.3274, 21.4999] # Corresponds to DOF[5]

SpeedUp1 = [Times1[0]/Times1[i] for i in range(1,len(Times1))] # Corresponds to DOF[0]
SpeedUp2 = [Times2[0]/Times2[i] for i in range(1,len(Times2))] # Corresponds to DOF[1]
SpeedUp3 = [Times3[0]/Times3[i] for i in range(1,len(Times3))] # Corresponds to DOF[2]
SpeedUp4 = [Times4[0]/Times4[i] for i in range(1,len(Times4))] # Corresponds to DOF[3]
SpeedUp5 = [Times5[0]/Times5[i] for i in range(1,len(Times5))] # Corresponds to DOF[4]
SpeedUp6 = [Times6[0]/Times6[i] for i in range(1,len(Times6))] # Corresponds to DOF[5]

WeakScalingSpeedUp = [Times1[0]/Times1[0], SpeedUp2[0], SpeedUp3[1], SpeedUp4[2], SpeedUp5[3], SpeedUp6[4]]

fig1 = plt.figure()
plt.plot(Ranks, Times1, color="royalblue", marker="o", label="17 Million DOF")
plt.plot(Ranks, Times2, color="firebrick", marker="^", label="34 Million DOF")
plt.plot(Ranks, Times3, color="forestgreen", marker="s", label="51 Million DOF")
plt.plot(Ranks, Times4, color="peru", marker="8", label="68 Million DOF")
plt.plot(Ranks, Times5, color="mediumpurple", marker="h", label="85 Million DOF")
plt.plot(Ranks, Times6, color="navy", marker="d", label="102 Million DOF")
plt.legend(ncol=2, fontsize=14)
plt.xlabel("Number Of MPI Ranks", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Time to Write File (seconds)", fontsize=14)
plt.ylim([3., 38.])
plt.yticks(fontsize=14)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/StrongScalingMPITimes.png", dpi=300)
plt.close(fig1)

fig2 = plt.figure()
plt.plot(Ranks[1:], SpeedUp1, color="royalblue", marker="o", label="17 Million DOF")
plt.plot(Ranks[1:], SpeedUp2, color="firebrick", marker="^", label="34 Million DOF")
plt.plot(Ranks[1:], SpeedUp3, color="forestgreen", marker="s", label="51 Million DOF")
plt.plot(Ranks[1:], SpeedUp4, color="peru", marker="8", label="68 Million DOF")
plt.plot(Ranks[1:], SpeedUp5, color="mediumpurple", marker="h", label="85 Million DOF")
plt.plot(Ranks[1:], SpeedUp6, color="navy", marker="d", label="102 Million DOF")
plt.legend(ncol=2, fontsize=14)
plt.xlabel("Number of MPI Ranks",fontsize=14)
plt.xticks([2, 3, 4, 5, 6], fontsize=14)
plt.ylabel("Speed Up", fontsize=14)
plt.ylim([1.05, 1.6])
plt.yticks(fontsize=14)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/StrongScalingMPISpeedup.png", dpi=300)
plt.close(fig2)

fig3 = plt.figure()
plt.plot(Ranks, WeakScalingSpeedUp, color="royalblue", marker="o", label="Weak Sacaling")
plt.legend(fontsize=14)
plt.xlabel("Number of MPI Ranks", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Speed Up", fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/WeakScalingMPITimes.png", dpi=300)
plt.close(fig3)






