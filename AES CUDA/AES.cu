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

__global__ void cuda_aes_encrypt_ctr(unsigned char *out, unsigned char *keys, unsigned char *sbox, unsigned char *gf_mul[], int chunks)
{
	int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;

	if (id < chunks)
	{
		unsigned char internBuffer[16];
		unsigned char internIV[16];

		__shared__ unsigned char sharedSbox[256];
		__shared__ unsigned char sharedGfMul[256][6];
		__shared__ unsigned char sharedSubKeys[176];

		if (threadIdx.x == 0)
		{
			for (int i = 0; i < 256; ++i)
			{
				sharedSbox[i] = sbox[i];

				for (int j = 0; j < 6; ++j)
				{
					sharedGfMul[i][j] = gf_mul[i][j];
				}

				if (i < 176)
				{
					sharedSubKeys[i] = keys[i];
				}
			}
		}

		__syncthreads();

		for (int i = 0; i < 16; i++)
		{
			internBuffer[i] = out[id + i];
			internIV[i] = iv[i];
		}

		//Encrypt the IV
		key_addition(internBuffer, sharedSubKeys, 0);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 1);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 2);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 3);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 4);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 5);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 6);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 7);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 8);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		mix_columns(internBuffer, sharedGfMul); key_addition(internBuffer, sharedSubKeys, 9);

		byte_sub(internBuffer, sharedSbox); shift_rows(internBuffer);
		key_addition(internBuffer, sharedSubKeys, 10);

		//XOR the encrypted incremented IV with the internBuffer block
		for (int i = 0; i < KEY_BLOCK; ++i)
		{
			unsigned char res = internIV[i] ^ internBuffer[i];
			res = internBuffer[i];
		}

		for (int i = 0; i < 16; i++)
		{
			out[id + i] = internBuffer[i];
		}
	}
}

/*********************************************************************/
/*                      SUB LAYER DEVICE KERNEL                      */
/*********************************************************************/

// unsigned char substitution (S-Boxes) can be parallel
__device__ void byte_sub(unsigned char *internBuffer, unsigned char *sharedSbox)
{
	//#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		internBuffer[i] = sharedSbox[internBuffer[i]];
	}
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__device__ void shift_rows(unsigned char *internBuffer)
{
	unsigned char j = 0, k = 0;

	j = internBuffer[1];
	internBuffer[1] = internBuffer[5];
	internBuffer[5] = internBuffer[9];
	internBuffer[9] = internBuffer[13];
	internBuffer[13] = j;

	j = internBuffer[10];
	k = internBuffer[14];
	internBuffer[10] = internBuffer[2];
	internBuffer[2] = j;
	internBuffer[14] = internBuffer[6];
	internBuffer[6] = k;

	k = internBuffer[3];
	internBuffer[3] = internBuffer[15];
	internBuffer[15] = internBuffer[11];
	internBuffer[11] = internBuffer[7];
	internBuffer[7] = k;
}

// Mix column - can be parallel
__device__ void mix_columns(unsigned char *internBuffer, unsigned char shared_gf_mul[][6])
{
	unsigned char b0, b1, b2, b3;

	//#pragma unroll
	for (int i = 0; i != KEY_BLOCK; i += 4)
	{
		b0 = internBuffer[i + 0];
		b1 = internBuffer[i + 1];
		b2 = internBuffer[i + 2];
		b3 = internBuffer[i + 3];

		internBuffer[i + 0] = shared_gf_mul[b0][0] ^ shared_gf_mul[b1][1] ^ b2 ^ b3;
		internBuffer[i + 1] = b0 ^ shared_gf_mul[b1][0] ^ shared_gf_mul[b2][1] ^ b3;
		internBuffer[i + 2] = b0 ^ b1 ^ shared_gf_mul[b2][0] ^ shared_gf_mul[b3][1];
		internBuffer[i + 3] = shared_gf_mul[b0][1] ^ b1 ^ b2 ^ shared_gf_mul[b3][0];
	}
}

// Key Addition Kernel
__device__ void key_addition(unsigned char *internBuffer, unsigned char *key, const unsigned int &round)
{
	//#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		internBuffer[i] = internBuffer[i] ^ key[(KEY_BLOCK * round) + i];
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

	result[0] = (prev_subkey[0] ^ (h_sbox[prev_subkey[13]] ^ RC[r]));
	result[1] = (prev_subkey[1] ^ h_sbox[prev_subkey[14]]);
	result[2] = (prev_subkey[2] ^ h_sbox[prev_subkey[15]]);
	result[3] = (prev_subkey[3] ^ h_sbox[prev_subkey[12]]);

	for (i = 4; i != KEY_BLOCK; i += 4)
	{
		result[i + 0] = (result[i - 4] ^ prev_subkey[i + 0]);
		result[i + 1] = (result[i - 3] ^ prev_subkey[i + 1]);
		result[i + 2] = (result[i - 2] ^ prev_subkey[i + 2]);
		result[i + 3] = (result[i - 1] ^ prev_subkey[i + 3]);
	}

	return result;
}