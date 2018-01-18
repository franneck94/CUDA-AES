/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <string>
#include <tuple>

#include "Helper.h"
#include "AES.h"
#include "timer.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*********************************************************************/
/*                       COUNTER MODE FUNCTIONS                      */
/*********************************************************************/

void counter_launch_kernel(unsigned char *messages, unsigned char *results, 
							unsigned char *key, unsigned char *IV, const unsigned int &message_size)
{
	int n = 0;
	int counter = 0;
	float milliseconds = 0.0f;

	unsigned char *keys = key_schedule(key);

	unsigned char *d_keys;
	unsigned char *d_messages;
	unsigned char *d_results;

	const int size_keys = NUM_ROUNDS * KEY_BLOCK * sizeof(unsigned char);
	const int size_messages = message_size * KEY_BLOCK * sizeof(unsigned char);
	const int size_results = message_size * KEY_BLOCK * sizeof(unsigned char);

	d_keys = new unsigned char[size_keys];
	d_messages = new unsigned char[size_messages];
	d_results = new unsigned char[size_results];

	gpuErrchk(cudaMalloc((void **)&d_keys, size_keys));
	gpuErrchk(cudaMalloc((void **)&d_messages, size_messages));
	gpuErrchk(cudaMalloc((void **)&d_results, size_results));

	gpuErrchk(cudaMemcpy(d_results, results, size_results, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_messages, messages, size_messages, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_keys, keys, size_keys, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(d_keySchedule, &d_keys, sizeof(size_keys)));

	GpuTimer timer;
	timer.Start();
	cuda_aes_encrypt_ctr << <BLOCKS_PER_LAUNCH, THREADS_PER_BLOCK >> > (d_messages, d_results, n, counter);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	timer.Stop();
	milliseconds = timer.ElapsedMilliSeconds();
	cout << "Done Counter Mode in: " << milliseconds << " (ms)." << endl;

	cudaMemcpy(messages, d_messages, size_messages, cudaMemcpyDeviceToHost);
	cudaMemcpy(results, d_results, size_results, cudaMemcpyDeviceToHost);

	cudaFree(keys);
	cudaFree(d_messages);
	cudaFree(d_results);
}

/*********************************************************************/
/*                        MAIN FUNCTION CALL                         */
/*********************************************************************/

int main()
{
	// Define Variables
	unsigned int iv_length = 12;

	string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
	string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/test.txt";;

	// Load data from files
	unsigned char *key = read_key(file_path_key);
	unsigned char *IV = random_byte_array(iv_length);

	cout << endl << "Starting AES CUDA - COUNTER MODE, with Key: " << endl;
	print_byte_array(key);

	std::tuple<unsigned char*, size_t> t = read_datafile(file_path_messages);
	unsigned char *messages = std::get<0>(t);
	size_t message_size = std::get<1>(t) * KEY_BLOCK;

	// Malloc Memory for Enc/Decrypted Solutions
	unsigned char *decrypted_solution;
	unsigned char *encrypted_solution;

	decrypted_solution = new unsigned char[message_size];
	encrypted_solution = new unsigned char[message_size];

	for (int i = 0; i != message_size; ++i)
	{
		decrypted_solution[i] = 0x00;
		encrypted_solution[i] = 0x00;
	}

	// Starting Encryption
	cout << endl << "Starting AES CUDA - COUNTER MODE KERNEL " << endl;
	counter_launch_kernel(messages, encrypted_solution, key, IV, message_size);

	// Starting Decryption
	cout << endl << "Starting AES CUDA - INVERSE COUNTER MODE KERNEL " << endl;
	counter_launch_kernel(encrypted_solution, decrypted_solution, key, IV, message_size);

	cout << endl << "Legit solution: " << check_byte_arrays(messages, decrypted_solution) << endl;

	getchar();
}