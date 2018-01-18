import matplotlib.pyplot as plt

test_sizes = [7, 11, 23, 40, 58, 78]
serial = [231, 354, 704, 1205, 1766, 2349]
openmp_best = [55, 66, 131, 221, 322, 414]
cuda_best = [0.472064, 0.62976, 1.19296, 2.01933, 3.3495, 3.86048]

plt.plot(test_sizes, serial, color="black")
plt.plot(test_sizes, openmp_best, color="blue")
plt.plot(test_sizes, cuda_best, color="red")

plt.legend(['Serial', 'OpenMP', 'CUDA'])
plt.title('Comparison')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig1.png")
plt.show()

test_sizes = [7, 11, 23, 40, 58, 78]
two = [117, 191, 375, 639, 959, 1223]
four = [61, 132, 209, 342, 498, 659]
six = [67, 81, 169, 276, 465, 525]
eight = [61, 75, 163, 257, 370, 500]
ten = [55, 66, 131, 221, 322, 414]
twelfe = [52, 69, 117, 196, 290, 386]

plt.plot(test_sizes, two, color="black")
plt.plot(test_sizes, four, color="blue")
plt.plot(test_sizes, six, color="yellow")
plt.plot(test_sizes, eight, color="red")
plt.plot(test_sizes, ten, color="orange")
plt.plot(test_sizes, twelfe, color="pink")

plt.legend(['2 Threads', '4 Threads', '6 Threads', '8 Threads', '10 Threads', '12 Threads'])
plt.title('OpenMP Comparison')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig2.png")
plt.show()

test_sizes = [7, 11, 23, 40, 58, 78]
two = [0.67584, 1.16941, 2.29274, 3.47546, 5.02477, 6.66522]
four = [0.512, 0.636928, 1.19706, 2.0439, 2.94298, 3.9465]
eight = [0.472064, 0.62976, 1.19296, 2.01933, 3.3495, 3.86048]

plt.plot(test_sizes, two, color="black")
plt.plot(test_sizes, four, color="blue")
plt.plot(test_sizes, eight, color="red")

plt.legend(['256 Threads', '512 Threads', '1024 Threads'])
plt.title('CUDA Comparison (Threads/Block)')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration (ms)')

plt.savefig("fig3.png")
plt.show()