import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

test_sizes = [7, 11, 23, 40, 58, 78]

######## BEST ##########

serial = [229, 358, 706, 1211, 1771, 2344]
openmp_best = [36, 57, 116, 197, 287, 381] # 12 Threads
cuda_best = [0.404378, 0.645216, 1.19008, 2.07104, 2.90038, 3.86745] # 1024 Threads

plt.plot(test_sizes, serial, color="black")
plt.plot(test_sizes, openmp_best, color="blue")
plt.plot(test_sizes, cuda_best, color="red")

plt.legend(['Serial', 'OpenMP (12 Threads)', 'CUDA (1024 Threads)'], prop=fontP)
plt.title('Comparison of Serial vs. OpenMP vs. CUDA')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig1.png")
plt.show()

######## OPENMP ##########

_2 = [118, 182, 367, 612, 898, 1181]
_4 = [68, 95, 192, 329, 475, 594]
_6 = [50, 80, 158, 264, 370, 487]
_8 = [49, 78, 152, 262, 371, 486]
_10 = [41, 65, 133, 223, 331, 434]
_12 = [36, 57, 116, 197, 287, 381]
_14 = [53, 74, 133, 219, 309, 414]
_16 = [51, 71, 131, 216, 312, 414]
_18 = [45, 67, 132, 207, 304, 417]

plt.plot(test_sizes, _2, color="black")
plt.plot(test_sizes, _4, color="brown")
plt.plot(test_sizes, _6, color="blue")
plt.plot(test_sizes, _8, color="pink")
plt.plot(test_sizes, _10, color="cyan")
plt.plot(test_sizes, _12, color="orange")
plt.plot(test_sizes, _14, color="red")
plt.plot(test_sizes, _16, color="yellow")
plt.plot(test_sizes, _18, color="green")

plt.legend(['2 Threads', '4 Threads', '6 Threads', '8 Threads', '10 Threads', '12 Threads', '14 Threads', '15 Threads', '18 Threads'], prop=fontP)
plt.title('OpenMP Comparison')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig2.png")
plt.show()

######## CUDA ##########

_128 = [1.35926, 2.06858, 4.08821, 6.84288, 9.31779, 13.5242]
_256 = [0.697955, 1.02881, 2.10596, 3.52799, 4.64189, 6.87462]
_512 = [0.423213, 0.651469, 1.20618, 2.08589, 3.02255, 4.00558]
_1024 = [0.404378, 0.645216, 1.19008, 2.07104, 2.90038, 3.86745]

plt.plot(test_sizes, _128, color="green")
plt.plot(test_sizes, _256, color="red")
plt.plot(test_sizes, _512, color="blue")
plt.plot(test_sizes, _1024, color="black")

plt.legend(['128 Threads', '256 Threads', '512 Threads', '1024 Threads'], prop=fontP)
plt.title('CUDA Comparison (Threads/Block)')
plt.xlabel('Test Size (MB)')
plt.ylabel('Duration of Encryption (ms)')

plt.savefig("fig3.png")
plt.show()