import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

test_sizes = [7, 11, 23, 40, 58, 78]

######## BEST ##########

serial = [229, 358, 706, 1211, 1771, 2344]
openmp_best = [36, 57, 116, 197, 287, 381] # 12 Threads
cuda_best = [1.6514, 2.45852, 4.73334, 7.96846, 9.8562, 13.009] # 1024 Threads

plt.plot(test_sizes, serial, color="black")
plt.plot(test_sizes, openmp_best, color="blue")
plt.plot(test_sizes, cuda_best, color="red")

plt.legend(['Serial', 'OpenMP (12 Threads)', 'CUDA (1024 Threads)'], prop=fontP, loc=2)
plt.title('Comparison of Serial vs. OpenMP vs. CUDA\nCPU: i7-8700k and GeForce GTX 1060')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig1.png")
plt.show()

######## OPENMP 2 - 8 ##########

_2 = [118, 182, 367, 612, 898, 1181]
_4 = [68, 95, 192, 329, 475, 594]
_6 = [50, 80, 158, 264, 370, 487]
_8 = [49, 78, 152, 262, 371, 486]

plt.plot(test_sizes, _2, color="black")
plt.plot(test_sizes, _4, color="brown")
plt.plot(test_sizes, _6, color="blue")
plt.plot(test_sizes, _8, color="green")

plt.legend(['2 Threads', '4 Threads', '6 Threads', '8 Threads'], prop=fontP, loc=2)
plt.title('OpenMP Comparison\nCPU: i7-8700k and GeForce GTX 1060')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig2.png")
plt.show()

######## OPENMP 10 - 18 ##########

_10 = [41, 65, 133, 223, 331, 434]
_12 = [36, 57, 116, 197, 287, 381]
_14 = [53, 74, 133, 219, 309, 414]
_16 = [51, 71, 131, 216, 312, 414]
_18 = [45, 67, 132, 207, 304, 417]

plt.plot(test_sizes, _10, color="cyan")
plt.plot(test_sizes, _12, color="orange")
plt.plot(test_sizes, _14, color="red")
plt.plot(test_sizes, _16, color="yellow")
plt.plot(test_sizes, _18, color="green")

plt.legend(['10 Threads', '12 Threads', '14 Threads', '15 Threads', '18 Threads'], prop=fontP, loc=2)
plt.title('OpenMP Comparison\nCPU: i7-8700k and GeForce GTX 1060')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig3.png")
plt.show()

######## CUDA Without Shared ##########

_512_without = [6.106, 9.427, 18.735, 26.117, 37.787, 55.945]
_128 = [2.16699, 3.46552, 6.52227, 11.6418, 15.9991, 19.9368]
_256 = [1.87207, 2.56041, 5.02825, 8.20777, 11.7854, 15.5998]
_512 = [1.63174, 2.48218, 4.75925, 7.95517, 11.4969, 14.8531]
_1024 = [1.6514, 2.45852, 4.73334, 7.96846, 9.8562, 13.009]

plt.plot(test_sizes, _512_without, color="orange")
plt.plot(test_sizes, _128, color="green")
plt.plot(test_sizes, _256, color="red")
plt.plot(test_sizes, _512, color="blue")
plt.plot(test_sizes, _1024, color="black")

plt.legend(['512 (Without)', '128 Threads', '256 Threads', '512 Threads', '1024 Threads'], prop=fontP, loc=2)
plt.title('CUDA Comparison (Threads/Block)\nCPU: i7-8700k and GeForce GTX 1060')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig4.png")
plt.show()