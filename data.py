import matplotlib.pyplot as plt

test_sizes = [7, 11, 23, 40, 58]
serial = [355, 534, 1088, 1830, 2653]
openmp_best = [46, 68, 132, 222, 322]
cuda_best = [0.834400, 1.262720, 2.447360, 4.311680, 6.031840]

plt.plot(test_sizes, serial, color="black")
plt.plot(test_sizes, openmp_best, color="blue")
plt.plot(test_sizes, cuda_best, color="red")

plt.legend(['Serial', 'OpenMP', 'CUDA'])
plt.title('Comparison')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig1.png")
plt.show()

test_sizes = [7, 11, 23, 40, 58]
two = [117, 191, 375, 639, 959]
four = [61, 132, 209, 342, 498]
six = [67, 83, 169, 276, 465]
eight = [55, 84, 160, 257, 360]
ten = [46, 68, 132, 222, 322]
sixteen = [55, 80, 134, 214, 324]

plt.plot(test_sizes, two, color="black")
plt.plot(test_sizes, four, color="blue")
plt.plot(test_sizes, six, color="yellow")
plt.plot(test_sizes, eight, color="red")
plt.plot(test_sizes, ten, color="orange")
plt.plot(test_sizes, sixteen, color="pink")

plt.legend(['2 Threads', '4 Threads', '6 Threads', '8 Threads', '10 Threads', '16 Threads'])
plt.title('OpenMP Comparison')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig2.png")
plt.show()

test_sizes = [7, 11, 23, 40, 58]
two = [0.937952, 1.398272, 2.741248, 4.791712, 6.740384]
four = [0.834400, 1.262720, 2.447360, 4.311680, 6.031840]
eight = [0.891680, 1.378176, 2.713472, 4.760992, 6.849312]

plt.plot(test_sizes, two, color="black")
plt.plot(test_sizes, four, color="blue")
plt.plot(test_sizes, eight, color="red")

plt.legend(['256 Threads', '512 Threads', '1024 Threads'])
plt.title('CUDA Comparison (Threads/Block)')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig3.png")
plt.show()