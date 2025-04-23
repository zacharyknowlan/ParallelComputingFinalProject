import matplotlib.pyplot as plt

# Data for MPI VTKWriter
WriteDOF = [17062806, 34028006, 51138918, 68211558, 85136918, 102278262]
Ranks = [1, 2, 3, 4, 5, 6]

Times1 = [5.36227, 4.60392, 4.21676, 4.50057, 4.56992, 4.77939] # Corresponds to DOF[0]
Times2 = [11.2400, 9.47284, 8.49174, 8.68912, 8.36811, 8.54736] # Corresponds to DOF[1]
Times3 = [15.4770, 13.4667, 11.7430, 12.1529, 11.6964, 11.6211] # Corresponds to DOF[2]
Times4 = [20.9097, 19.1745, 15.4112, 15.1178, 15.4682, 15.0777] # Corresponds to DOF[3]
Times5 = [24.7205, 20.3123, 17.8193, 18.6405, 17.3797, 18.2512] # Corresponds to DOF[4]
Times6 = [30.8935, 25.7645, 22.8770, 21.9028, 21.3274, 21.4999] # Corresponds to DOF[5]

# Strong scaling speedup for MPI VTKWriter
SpeedUp1 = [Times1[0]/Times1[i] for i in range(1,len(Times1))] # Corresponds to DOF[0]
SpeedUp2 = [Times2[0]/Times2[i] for i in range(1,len(Times2))] # Corresponds to DOF[1]
SpeedUp3 = [Times3[0]/Times3[i] for i in range(1,len(Times3))] # Corresponds to DOF[2]
SpeedUp4 = [Times4[0]/Times4[i] for i in range(1,len(Times4))] # Corresponds to DOF[3]
SpeedUp5 = [Times5[0]/Times5[i] for i in range(1,len(Times5))] # Corresponds to DOF[4]
SpeedUp6 = [Times6[0]/Times6[i] for i in range(1,len(Times6))] # Corresponds to DOF[5]

# Weak scaling speedup for MPI VTKWriter
WeakScalingSpeedUp = [Times1[0]/Times1[0], SpeedUp2[0], SpeedUp3[1], SpeedUp4[2], SpeedUp5[3], SpeedUp6[4]]
Description = ["1 Rank,\n 17M DOF", "2 Ranks,\n 34M DOF", "3 Ranks,\n 51M DOF", 
                "4 Ranks,\n 68M DOF", "5 Ranks,\n 85M DOF", "6 Ranks,\n 102M DOF"]
Colors = ["royalblue", "firebrick", "forestgreen", "peru", "mediumpurple", "navy"]

# Data for CUDA partial assembly
SerialSolver = [3.07996, 44.0438, 131.979, 1617.83, 3913.09]
SerialTotal = [3.13574, 44.115, 132.119, 1618.44, 3914.31]
CUDA64Solver = [0.474353, 3.19875, 8.25247, 105.169, 303.853]
CUDA64Total = [0.481736, 3.22079, 8.28954, 105.377, 304.252]
CUDA128Solver = [0.456072, 2.98565, 7.36688, 104.042, 285.133]
CUDA128Total = [0.46444, 3.00822, 7.4027, 104.207, 285.46]
CUDA256Solver = [0.453173, 2.88988, 7.38826, 102.488, 275.134]
CUDA256Total = [0.460059, 2.91283, 7.42481, 102.658, 275.527]
CUDA512Solver = [0.444567, 2.85578, 7.24109, 97.5009, 289.253]
CUDA512Total = [0.451626, 3.15433, 7.28012, 97.6672, 289.828]

# Speedup for CUDA partial assembly with various block sizes
SolveDOF = [35718, 178278, 342806, 1712262, 3447126]
SpeedUpCUDA64Solver = [SerialSolver[i]/CUDA64Solver[i] for i in range(0,len(SerialSolver))]
SpeedUpCUDA128Solver = [SerialSolver[i]/CUDA128Solver[i] for i in range(0,len(SerialSolver))]
SpeedUpCUDA256Solver = [SerialSolver[i]/CUDA256Solver[i] for i in range(0,len(SerialSolver))]
SpeedUpCUDA512Solver = [SerialSolver[i]/CUDA512Solver[i] for i in range(0,len(SerialSolver))]

# 4 element mesh serial and CUDA comparison to assess overhead
SmallMeshSolver = [0.00332643, 0.0603897, 0.0600934, 0.0590039, 0.0599821]
SmallMeshTotal = [0.049087, 0.0655181, 0.0649574, 0.0628865, 0.0649226]
Description2 = ["Serial", "CUDA,\n BlockSize 64", "CUDA,\n BlockSize 128", 
                "CUDA,\n BlockSize 256", "CUDA,\n BlockSize 512"]
Colors2 = ["royalblue", "firebrick", "peru","mediumpurple", "forestgreen"]

# Percentage of time spent in solver for CUDA partial assembly with block size of 256
PercentageCUDA256 = [100.*CUDA64Solver[i]/CUDA64Total[i] for i in range(0,len(CUDA64Total))]
Description3 = ["36k DOF", "178k DOF", "342k DOF", "1.7M DOF", "3.4M DOF"]
Colors3 = ["mediumpurple", "peru", "royalblue", "firebrick", "navy"]

# All figures used in report
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

fig3 = plt.figure(figsize=[7,5.25])
plt.bar(Description, WeakScalingSpeedUp, color=Colors)
plt.ylim([0.9, 1.5])
plt.yticks(fontsize=14)
plt.ylabel("Speed Up", fontsize=14)
plt.xticks(fontsize=14)
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/WeakScalingMPITimes.png", dpi=300)
plt.close(fig3)

fig4 = plt.figure(figsize=[9,5])
plt.bar(Description2, SmallMeshSolver, color=Colors2)
plt.ylim([0., 0.07])
plt.yticks(fontsize=14)
plt.ylabel("Solver Time (seconds)", fontsize=14)
plt.xticks(fontsize=14)
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/SmallMeshOverhead.png", dpi=300)
plt.close(fig4)

fig5 = plt.figure()
plt.plot(SolveDOF, SpeedUpCUDA64Solver, color="royalblue", marker="o", label="BlockSize = 64")
plt.plot(SolveDOF, SpeedUpCUDA128Solver, color="peru", marker="s", label="BlockSize = 128")
plt.plot(SolveDOF, SpeedUpCUDA256Solver, color="mediumpurple", marker="^", label="BlockSize = 256")
plt.plot(SolveDOF, SpeedUpCUDA512Solver, color="firebrick", marker="8", label="BlockSize = 512")
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Problem Size (DOF)", fontsize=14)
plt.xscale("log")
plt.yticks(fontsize=14)
plt.ylabel("Speed Up", fontsize=14)
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/CUDABlockSizeSpeedUp.png", dpi=300)
plt.close(fig5)

fig6 = plt.figure()
plt.bar(Description3, PercentageCUDA256, color=Colors3)
plt.xticks(fontsize=14)
plt.yticks([98, 98.5, 99, 99.5, 100], fontsize=14)
plt.ylabel("Percentage of Runtime in Solver", fontsize=14)
plt.ylim([98, 100])
plt.tight_layout(pad=0.8)
plt.savefig("/lore/knowlz/ParallelComputingProject/Plots/SolverPercentageOfTime.png", dpi=300)
plt.close(fig6)
