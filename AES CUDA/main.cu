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

#include "Helper.h"
#include "AES.h"
#include "timer.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

#define THREADS_PER_BLOCK 1024
#define ROUNDS 10
#define SHARED false

/*********************************************************************/
/*                       GPU HELPER FUNCTIONS                        */
/*********************************************************************/

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

/*********************************************************************/
/*                       COUNTER MODE FUNCTIONS                      */
/*********************************************************************/

void counter_launch_kernel(unsigned char *messages, unsigned char *results, unsigned char *keys,
							const unsigned int &message_size, const unsigned int &filesize)
{
	float milliseconds = 0.0f;

	// Define launch config
	int chunks = filesize / KEY_BLOCK;
	int ThreadsPerBlock = THREADS_PER_BLOCK;
	int Blocks = ceil(chunks / ThreadsPerBlock);

	// Messages to device memory
	unsigned char *d_messages;
	gpuErrchk(cudaMalloc((void **)&d_messages, message_size * sizeof(unsigned char)));
	gpuErrchk(cudaMemcpy(d_messages, messages, message_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Results to device memory
	unsigned char *d_results;
	gpuErrchk(cudaMalloc((void **)&d_results, message_size * sizeof(unsigned char)));
	gpuErrchk(cudaMemcpy(d_results, messages, message_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// SBOX to device memory
	unsigned char *d_sbox;
	if (SHARED == true)
	{
		gpuErrchk(cudaMalloc((void **)&d_sbox, 256 * sizeof(unsigned char)));
		gpuErrchk(cudaMemcpy(d_sbox, h_sbox, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}

	// Subkeys to device memory
	unsigned char *d_keys;
	gpuErrchk(cudaMalloc((void **)&d_keys, NUM_ROUNDS * KEY_BLOCK * sizeof(unsigned char)));
	gpuErrchk(cudaMemcpy(d_keys, keys, NUM_ROUNDS * KEY_BLOCK * sizeof(unsigned char), cudaMemcpyHostToDevice));

	for (int i = 0; i != ROUNDS; ++i)
	{
		GpuTimer timer;
		timer.Start();

		if (SHARED == true)
		{
			aes_encryption_shared << <Blocks, ThreadsPerBlock >> > (d_messages, d_results, d_sbox, d_keys, message_size);
		}
		else
		{
			aes_encryption << <Blocks, ThreadsPerBlock >> > (d_messages, d_results, d_keys, message_size);
		}

		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		timer.Stop();
		milliseconds += timer.ElapsedMilliSeconds();
	}

	cout << "Done Counter Mode in: " << milliseconds / (float) NUM_ROUNDS << " (ms)." << endl;

	gpuErrchk(cudaMemcpy(results, d_results, message_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_results));
	gpuErrchk(cudaFree(d_messages));
	gpuErrchk(cudaFree(d_keys));

	if (SHARED == true)
	{
		gpuErrchk(cudaFree(d_sbox));
	}
}

/*********************************************************************/
/*                        MAIN FUNCTION CALL                         */
/*********************************************************************/

int main()
{
	for (int i = 6; i > 0; i--)
	{
		cout << endl << "Text" << i;
		string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
		string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/text" + std::to_string(i) + ".txt";
		int filesize = file_size(file_path_messages.c_str());

		// Load data from files
		unsigned char *key = read_key(file_path_key);
		unsigned char *keys = key_schedule(key);

		cout << endl << std::dec << "Starting AES CUDA - COUNTER MODE" << endl;

		// Read in data
		unsigned char * plaintexts = (unsigned char *)malloc(sizeof(unsigned char)*filesize);
		read_datafile(file_path_messages.c_str(), plaintexts);

		// Malloc Memory for Enc/Decrypted Solutions
		unsigned char *decrypted_solution;
		unsigned char *encrypted_solution;

		decrypted_solution = new unsigned char[filesize];
		encrypted_solution = new unsigned char[filesize];

		cout << endl << "Ready to start!" << endl << endl;

		// Starting Encryption
		cout << endl << "Starting AES CUDA - COUNTER MODE KERNEL " << endl;
		counter_launch_kernel(plaintexts, encrypted_solution, keys, filesize, filesize);

		// Starting Decryption
		cout << endl << "Starting AES CUDA - INVERSE COUNTER MODE KERNEL " << endl;
		counter_launch_kernel(encrypted_solution, decrypted_solution, keys, filesize, filesize);

		// Checking if Decryption of Encryption is the plaintext
		cout << endl << "Legit solution: " << check_byte_arrays(plaintexts, decrypted_solution, filesize) << endl;
	}

	getchar();
}