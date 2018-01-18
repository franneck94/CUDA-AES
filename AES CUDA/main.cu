/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <fstream>
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
		getchar();
		if (abort) exit(code);
	}
}

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig)
{
	std::vector<T> ret;
	for (const auto &v : orig)
		ret.insert(ret.end(), v.begin(), v.end());
	return ret;
}

int file_size(const std::string &add) 
{
	ifstream mySource;
	mySource.open(add, std::ios_base::binary);
	mySource.seekg(0, std::ios_base::end);
	int size = mySource.tellg();
	mySource.close();
	return size;
}

/*********************************************************************/
/*                       COUNTER MODE FUNCTIONS                      */
/*********************************************************************/

void counter_launch_kernel(unsigned char *messages, unsigned char *results, unsigned char *key,
							const unsigned int &message_size, const unsigned int &filesize)
{
	float milliseconds = 0.0f;

	// Define launch config
	int chunks = filesize / 16;
	int ThreadsPerBlock = 1024;
	int Blocks = chunks / ThreadsPerBlock;

	// Pushes results to device memory
	unsigned char *d_results;
	int size_results = message_size * sizeof(unsigned char);
	gpuErrchk(cudaMalloc((void **)&d_results, size_results));
	gpuErrchk(cudaMemcpy(d_results, messages, size_results, cudaMemcpyHostToDevice));

	// SBOX to device memory
	unsigned char *d_sbox;
	int size_sbox = 256 * sizeof(unsigned char);
	gpuErrchk(cudaMalloc((void **)&d_sbox, size_sbox));
	gpuErrchk(cudaMemcpy(d_sbox, h_sbox, size_sbox, cudaMemcpyHostToDevice));

	// GFMul to device memory
	unsigned char **d_gfmul;
	int size_gfmul = 256 * 6 * sizeof(unsigned char);
	d_gfmul = new unsigned char*[size_gfmul];

	for (int i = 0; i != 256; ++i)
	{
		d_gfmul[i] = new unsigned char[6];
	}

	gpuErrchk(cudaMalloc((void **)&d_gfmul, size_gfmul));
	gpuErrchk(cudaMemcpy(d_gfmul, h_gfmul, size_gfmul, cudaMemcpyHostToDevice));

	// Subkeys to device memory
	unsigned char *keys = key_schedule(key);
	unsigned char *d_keys;
	const int size_keys = NUM_ROUNDS * KEY_BLOCK * sizeof(unsigned char);
	gpuErrchk(cudaMalloc((void **)&d_keys, size_keys));
	gpuErrchk(cudaMemcpy(d_keys, keys, size_keys, cudaMemcpyHostToDevice));

	GpuTimer timer;
	timer.Start();
	cuda_aes_encrypt_ctr << <Blocks, ThreadsPerBlock >> > (d_results, d_keys, d_sbox, d_gfmul, chunks);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	timer.Stop();
	milliseconds = timer.ElapsedMilliSeconds();
	cout << "Done Counter Mode in: " << milliseconds << " (ms)." << endl;

	cudaMemcpy(results, d_results, size_results, cudaMemcpyDeviceToHost);
	cudaFree(d_results);
}

/*********************************************************************/
/*                        MAIN FUNCTION CALL                         */
/*********************************************************************/

int main()
{
	string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
	string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/test.txt";
	int filesize = file_size(file_path_messages);

	// Load data from files
	unsigned char *key = read_key(file_path_key);

	cout << endl << std::dec << "Starting AES CUDA - COUNTER MODE, with Key: " << endl;
	print_byte_array(key, 16);
	cout << endl << "And Filesize: " << filesize << endl;

	vector<unsigned char> data_vec = read_datafile(file_path_messages);
	size_t message_size = data_vec.size();

	cout << endl << std::dec << data_vec.size() << endl;

	// Malloc Memory for Enc/Decrypted Solutions
	unsigned char *decrypted_solution;
	unsigned char *encrypted_solution;
	unsigned char *messages;

	messages = new unsigned char[message_size];
	decrypted_solution = new unsigned char[message_size];
	encrypted_solution = new unsigned char[message_size];

	for (int i = 0; i != message_size; ++i)
	{
		messages = &data_vec[i];
		decrypted_solution = 0x00;
		encrypted_solution = 0x00;
	}

	cout << endl << "Ready to start!" << endl << endl;

	// Starting Encryption
	getchar();
	cout << endl << "Starting AES CUDA - COUNTER MODE KERNEL " << endl;
	counter_launch_kernel(messages, encrypted_solution, key, message_size, filesize);

	cout << endl << "size = " << sizeof(encrypted_solution) << endl;

	// Starting Decryption
	getchar();
	cout << endl << "Starting AES CUDA - INVERSE COUNTER MODE KERNEL " << endl;
	counter_launch_kernel(encrypted_solution, decrypted_solution, key, message_size, filesize);

	cout << endl << "Legit solution: " << check_byte_arrays(messages, decrypted_solution, message_size) << endl;

	getchar();
}