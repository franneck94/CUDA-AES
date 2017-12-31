/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "AES.h"
#include "Helper.h"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                           CONSTRUCTORS                            */
/*********************************************************************/

// Constructor of AES en/decryption
AES::AES(unsigned char *key)
{
	m_subkeys = new unsigned char* [SUB_KEYS];
	m_key = key;
	key_schedule();
}

/*********************************************************************/
/*                           KEY FUNCTIONS                           */
/*********************************************************************/

// Computing the round keys
void AES::key_schedule()
{
	int r;

	for (r = 0; r != SUB_KEYS; r++)
	{
		if (r == 0)
			m_subkeys[r] = m_key;
		else
		{
			if (AES_BITS == 128)
				m_subkeys[r] = sub_key128(m_subkeys[r - 1], r - 1);
			else
				cout << "TODO! 192-bit and 256-bit not implemented yet." << endl;
		}
	}
}

// Computing subkeys for round 1 up to 10
unsigned char* AES::sub_key128(unsigned char *prev_subkey, const int &r)
{
	unsigned char *result;
	result = new unsigned char[KEY_BLOCK];

	int i;

	result[0] = (prev_subkey[0] ^ (sbox[prev_subkey[13]] ^ RC[r]));
	result[1] = (prev_subkey[1] ^ sbox[prev_subkey[14]]);
	result[2] = (prev_subkey[2] ^ sbox[prev_subkey[15]]);
	result[3] = (prev_subkey[3] ^ sbox[prev_subkey[12]]);

	for (i = 4; i != KEY_BLOCK; i += 4)
	{
		result[i + 0] = (result[i - 4] ^ prev_subkey[i + 0]);
		result[i + 1] = (result[i - 3] ^ prev_subkey[i + 1]);
		result[i + 2] = (result[i - 2] ^ prev_subkey[i + 2]);
		result[i + 3] = (result[i - 1] ^ prev_subkey[i + 3]);
	}

	return result;
}

unsigned char** AES::get_subkeys()
{
	return m_subkeys;
}

/*********************************************************************/
/*                       EN- DECRYPTION FUNCTIONS                    */
/*********************************************************************/

// Starting the encryption phase
void encrypt(unsigned char *message, unsigned char **subkeys)
{
	// CUDA Block Dimensions
	dim3 dim_block_key_addition(KEY_BLOCK);
	dim3 dim_block_byte_sub(KEY_BLOCK);
	dim3 dim_block_shift_row(SHIFT_ROW_LIMIT);
	dim3 dim_block_mix_column(MIX_COLUMN_LIMIT);

	int round = 0;

	// Allocate GPU Memory
	unsigned char *d_message = NULL;
	unsigned char **d_subkeys = NULL;
	cudaMalloc((void **)&d_message, sizeof(message));
	cudaMalloc((void **)&d_subkeys, sizeof(subkeys));
	cudaMemcpy(d_message, message, sizeof(message), cudaMemcpyHostToDevice);
	cudaMemcpy(d_subkeys, subkeys, sizeof(subkeys), cudaMemcpyHostToDevice);

	// Key-Add before round 1 (R0)
	key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);
	round = 1;

	// Round 1 to NUM_ROUNDS - 1 (R1 to R9)
	for (round; round != NUM_ROUNDS; round++)
	{
		byte_sub_kernel << <1, dim_block_byte_sub >> >(d_message);
		shift_rows_kernel << <1, dim_block_shift_row >> >(d_message);
		mix_columns_kernel << <1, dim_block_mix_column >> >(d_message);
		key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);
	}

	// Last round without Mix-Column (RNUM_ROUNDS)
	round = NUM_ROUNDS;
	byte_sub_kernel << <1, dim_block_byte_sub >> >(d_message);
	shift_rows_kernel << <1, dim_block_shift_row >> >(d_message);
	key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);

	// Deallocate GPU Memory
	cudaMemcpy(message, d_message, sizeof(message), cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_subkeys);
}

// Starting the decryption phase
void decrypt(unsigned char *message, unsigned char **subkeys)
{
	// CUDA Block Dimensions
	dim3 dim_block_key_addition(KEY_BLOCK);
	dim3 dim_block_byte_sub(KEY_BLOCK);
	dim3 dim_block_shift_row(SHIFT_ROW_LIMIT);
	dim3 dim_block_mix_column(MIX_COLUMN_LIMIT);

	int round = NUM_ROUNDS;

	// Allocate GPU Memory
	unsigned char *d_message = NULL;
	unsigned char **d_subkeys = NULL;
	cudaMalloc((void **)&d_message, sizeof(message));
	cudaMalloc((void **)&d_subkeys, sizeof(subkeys));
	cudaMemcpy(d_message, message, sizeof(message), cudaMemcpyHostToDevice);
	cudaMemcpy(d_subkeys, subkeys, sizeof(subkeys), cudaMemcpyHostToDevice);

	// Key-Add before round (Inverse NUM_ROUNDS)
	key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);
	shift_rows_inv_kernel << <1, dim_block_shift_row >> >(d_message);
	byte_sub_inv_kernel << <1, dim_block_byte_sub >> >(d_message);
	round = NUM_ROUNDS - 1;

	// Round NUM_ROUNDS - 10 to 1 (Inverse R9 to R1)
	for (round; round > 0; round--)
	{
		key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);
		mix_columns_inv_kernel << <1, dim_block_mix_column >> >(d_message);
		shift_rows_inv_kernel << <1, dim_block_shift_row >> >(d_message);
		byte_sub_inv_kernel << <1, dim_block_byte_sub >> >(d_message);
	}

	// Last round without Mix-Column (Inverse R0)
	round = 0;
	key_addition_kernel << <1, dim_block_key_addition >> >(d_message, d_subkeys, round);

	// Deallocate GPU Memory
	cudaMemcpy(message, d_message, sizeof(message), cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_subkeys);
}

/*********************************************************************/
/*                        SUB LAYER KERNEL                           */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
__global__ void byte_sub_kernel(unsigned char *message)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < KEY_BLOCK)
	{
		message[id] = d_sbox[message[id]];
	}
}

// Inverse byte substitution (S-Boxes) can be parallel
__global__ void byte_sub_inv_kernel(unsigned char *message)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < KEY_BLOCK)
	{
		message[id] = d_sboxinv[message[id]];
	}
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__global__ void shift_rows_kernel(unsigned char *message)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned char j = 0, k = 0;

	if (id < SHIFT_ROW_LIMIT)
	{
		if (id == 0)
		{
			j = message[1];
			message[1] = message[5];
			message[5] = message[9];
			message[9] = message[13];
			message[13] = j;
		}
		else if (id == 1)
		{
			j = message[10];
			k = message[14];
			message[10] = message[2];
			message[2] = j;
			message[14] = message[6];
			message[6] = k;
		}
		else
		{
			k = message[3];
			message[3] = message[15];
			message[15] = message[11];
			message[11] = message[7];
			message[7] = k;
		}
	}
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
__global__ void shift_rows_inv_kernel(unsigned char *message)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned char j = 0, k = 0;

	if (id < SHIFT_ROW_LIMIT)
	{
		if (id == 0)
		{
			j = message[1];
			message[1] = message[13];
			message[13] = message[9];
			message[9] = message[5];
			message[5] = j;
		}
		else if (id == 1)
		{
			j = message[2];
			k = message[6];
			message[2] = message[10];
			message[10] = j;
			message[6] = message[14];
			message[14] = k;
		}
		else
		{
			j = message[3];
			message[3] = message[7];
			message[7] = message[11];
			message[11] = message[15];
			message[15] = j;
		}
	}
}

// Mix column - can be parallel
__global__ void mix_columns_kernel(unsigned char *message)
{
	unsigned char b0, b1, b2, b3;
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < MIX_COLUMN_LIMIT)
	{
		b0 = message[id + 0];
		b1 = message[id + 1];
		b2 = message[id + 2];
		b3 = message[id + 3];

		// Mix-Col Matrix * b vector
		message[id + 0] = d_mul[b0][0] ^ d_mul[b1][1] ^ b2 ^ b3;
		message[id + 1] = b0 ^ d_mul[b1][0] ^ d_mul[b2][1] ^ b3;
		message[id + 2] = b0 ^ b1 ^ d_mul[b2][0] ^ d_mul[b3][1];
		message[id + 3] = d_mul[b0][1] ^ b1 ^ b2 ^ d_mul[b3][0];
	}
}

// Inverse mix column
__global__ void mix_columns_inv_kernel(unsigned char *message)
{
	unsigned char c0, c1, c2, c3;
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < MIX_COLUMN_LIMIT)
	{
		c0 = message[id + 0];
		c1 = message[id + 1];
		c2 = message[id + 2];
		c3 = message[id + 3];

		// Mix-Col Inverse Matrix * c vector
		message[id + 0] = d_mul[c0][5] ^ d_mul[c1][3] ^ d_mul[c2][4] ^ d_mul[c3][2];
		message[id + 1] = d_mul[c0][2] ^ d_mul[c1][5] ^ d_mul[c2][3] ^ d_mul[c3][4];
		message[id + 2] = d_mul[c0][4] ^ d_mul[c1][2] ^ d_mul[c2][5] ^ d_mul[c3][3];
		message[id + 3] = d_mul[c0][3] ^ d_mul[c1][4] ^ d_mul[c2][2] ^ d_mul[c3][5];
	}
}

// Key Addition Kernel
__global__ void key_addition_kernel(unsigned char *message, unsigned char **subkeys, const unsigned int &round)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < KEY_BLOCK)
	{
		message[id] = message[id] ^ subkeys[round][id];
	}
}