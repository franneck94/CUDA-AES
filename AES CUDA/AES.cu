/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "AES.h"
#include "Helper.h"

using std::cout;
using std::endl;
using std::vector;

__device__ unsigned char iv[KEY_BLOCK] = 
{ 
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 
};

/*********************************************************************/
/*                           MAIN KERNEL                             */
/*********************************************************************/

__global__ void cuda_aes_encrypt_ctr(unsigned char *out)
{
	int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;

	unsigned char internBuffer[16];
	unsigned char internIV[16];

	for (int i = 0; i < 16; i++) 
	{
		internBuffer[i] = out[id + i];
		internIV[i] = iv[i];
	}

	__syncthreads();

	//Encrypt the IV
	aes_encrypt(internIV, d_keySchedule);

	__syncthreads();

	//XOR the encrypted incremented IV with the message block
	for (int i = 0; i < KEY_BLOCK; ++i)
	{
		unsigned char res = internIV[i] ^ internBuffer[i];
		res = internBuffer[i];
	}

	__syncthreads();

	for (int i = 0; i < 16; i++) 
	{
		out[id + i] = internBuffer[i];
	}
}

/*********************************************************************/
/*                         MAIN DEVICE KERNEL                        */
/*********************************************************************/

__device__ void aes_encrypt(unsigned char out[], unsigned char *key)
{
	// Do all AES Rounds
	if (AES_BITS == 128)
	{
		key_addition(out, key, 0);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 1 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 2 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 3 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 4 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 5 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 6 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 7 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 8 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); mix_columns(out); key_addition(out, key, 9 * KEY_BLOCK);
		byte_sub(out); shift_rows(out); key_addition(out, key, 10 * KEY_BLOCK);
	}
}

/*********************************************************************/
/*                      SUB LAYER DEVICE KERNEL                      */
/*********************************************************************/

// unsigned char substitution (S-Boxes) can be parallel
__device__ void byte_sub(unsigned char message[])
{
	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		message[i] = d_sbox[message[i]];
	}
}

// Inverse unsigned char substitution (S-Boxes) can be parallel
__device__ void byte_sub_inv(unsigned char message[])
{
	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		message[i] = d_sboxinv[message[i]];
	}
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__device__ void shift_rows(unsigned char message[])
{
	unsigned char j = 0, k = 0;

	j = message[1];
	message[1] = message[5];
	message[5] = message[9];
	message[9] = message[13];
	message[13] = j;

	j = message[10];
	k = message[14];
	message[10] = message[2];
	message[2] = j;
	message[14] = message[6];
	message[6] = k;

	k = message[3];
	message[3] = message[15];
	message[15] = message[11];
	message[11] = message[7];
	message[7] = k;
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
__device__ void shift_rows_inv(unsigned char message[])
{
	unsigned char j = 0, k = 0;

	j = message[1];
	message[1] = message[13];
	message[13] = message[9];
	message[9] = message[5];
	message[5] = j;

	j = message[2];
	k = message[6];
	message[2] = message[10];
	message[10] = j;
	message[6] = message[14];
	message[14] = k;

	j = message[3];
	message[3] = message[7];
	message[7] = message[11];
	message[11] = message[15];
	message[15] = j;
}

// Mix column - can be parallel
__device__ void mix_columns(unsigned char message[])
{
	unsigned char b0, b1, b2, b3;

	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; i += 4)
	{
		b0 = message[i + 0];
		b1 = message[i + 1];
		b2 = message[i + 2];
		b3 = message[i + 3];

		message[i + 0] = d_mul[b0][0] ^ d_mul[b1][1] ^ b2 ^ b3;
		message[i + 1] = b0 ^ d_mul[b1][0] ^ d_mul[b2][1] ^ b3;
		message[i + 2] = b0 ^ b1 ^ d_mul[b2][0] ^ d_mul[b3][1];
		message[i + 3] = d_mul[b0][1] ^ b1 ^ b2 ^ d_mul[b3][0];
	}
}

// Inverse mix column
__device__ void mix_columns_inv(unsigned char message[])
{
	unsigned char c0, c1, c2, c3;

	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; i += 4)
	{
		c0 = message[i + 0];
		c1 = message[i + 1];
		c2 = message[i + 2];
		c3 = message[i + 3];

		message[i + 0] = d_mul[c0][5] ^ d_mul[c1][3] ^ d_mul[c2][4] ^ d_mul[c3][2];
		message[i + 1] = d_mul[c0][2] ^ d_mul[c1][5] ^ d_mul[c2][3] ^ d_mul[c3][4];
		message[i + 2] = d_mul[c0][4] ^ d_mul[c1][2] ^ d_mul[c2][5] ^ d_mul[c3][3];
		message[i + 3] = d_mul[c0][3] ^ d_mul[c1][4] ^ d_mul[c2][2] ^ d_mul[c3][5];
	}
}

// Key Addition Kernel
__device__ void key_addition(unsigned char message[], unsigned char *key, const unsigned int &start)
{
	#pragma unroll
	for (int i = start; i != start + KEY_BLOCK; ++i)
	{
		message[i] = message[i] ^ key[i];
	}
}

/*********************************************************************/
/*                           KEY FUNCTIONS                           */
/*********************************************************************/

// Computing the round keys
unsigned char* key_schedule(unsigned char *key)
{
	int r;
	unsigned char **subkeys;
	subkeys = new unsigned char*[NUM_ROUNDS];

	for (r = 0; r != SUB_KEYS; r++)
	{
		if (r == 0)
			subkeys[r] = key;
		else
		{
			if (AES_BITS == 128)
				subkeys[r] = sub_key128(subkeys[r - 1], r - 1);
			else
				cout << "TODO! 192-bit and 256-bit not implemented yet." << endl;
		}
	}

	unsigned char *keys;
	keys = new unsigned char[NUM_ROUNDS * KEY_BLOCK];

	for (int i = 0; i != NUM_ROUNDS; ++i)
	{
		for (int j = 0; j != KEY_BLOCK; ++j)
		{
			keys[i * NUM_ROUNDS + j] = subkeys[i][j];
		}
	}

	return keys;
}

// Computing subkeys for round 1 up to 10
unsigned char* sub_key128(unsigned char *prev_subkey, const int &r)
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