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

__device__ unsigned char IV[KEY_BLOCK] = 
{ 
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 
};

/*********************************************************************/
/*                           MAIN KERNEL                             */
/*********************************************************************/

__global__ void aes_encryption(unsigned char* SBOX, unsigned char* BufferData, unsigned char* SubKeys)
{
	register int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;

	__shared__ unsigned char sharedSbox[256];
	__shared__ unsigned char sharedSubKeys[176];

	if (threadIdx.x == 0)
	{
		for (int i = 0; i<256; i++)
		{
			sharedSbox[i] = SBOX[i];

			if (i < 176)
			{
				sharedSubKeys[i] = SubKeys[i];
			}
		}
	}

	__syncthreads();

	register unsigned char internBuffer[16];
	register unsigned char internIV[16];

	for (register int i = 0; i < 16; i++)
	{
		internBuffer[i] = BufferData[id + i];
		internIV[i] = IV[i];
	}

	key_addition(internIV, sharedSubKeys, 0);

	for (register int i = 1; i < 11; i++)
	{
		byte_sub(internIV, sharedSbox);
		shift_rows(internIV);

		if (i != 10)
		{
			mix_columns(internIV);
		}

		key_addition(internIV, sharedSubKeys, i);
	}

	//XOR the encrypted incremented IV with the internBuffer block
	for (int i = 0; i < KEY_BLOCK; ++i)
	{
		unsigned char res = internIV[i] ^ internBuffer[i];
		res = internBuffer[i];
	}

	for (int i = 0; i < 16; i++)
	{
		BufferData[id + i] = internBuffer[i];
	}
}

/*********************************************************************/
/*                      SUB LAYER DEVICE KERNEL                      */
/*********************************************************************/

// Key Addition Kernel
__device__ void key_addition(unsigned char *internBuffer, unsigned char *key, const unsigned int &round)
{
	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		internBuffer[i] = internBuffer[i] ^ key[(KEY_BLOCK * round) + i];
	}
}

// unsigned char substitution (S-Boxes) can be parallel
__device__ void byte_sub(unsigned char *internBuffer, unsigned char *sharedSbox)
{
	#pragma unroll
	for (int i = 0; i != KEY_BLOCK; ++i)
	{
		internBuffer[i] = sharedSbox[internBuffer[i]];
	}
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__device__ void shift_rows(unsigned char *internBuffer)
{
	unsigned char tmpBuffer;

	tmpBuffer = internBuffer[1];
	internBuffer[1] = internBuffer[5];
	internBuffer[5] = internBuffer[9];
	internBuffer[9] = internBuffer[13];
	internBuffer[13] = tmpBuffer;

	tmpBuffer = internBuffer[2];
	internBuffer[2] = internBuffer[10];
	internBuffer[10] = tmpBuffer;
	tmpBuffer = internBuffer[6];
	internBuffer[6] = internBuffer[14];
	internBuffer[14] = tmpBuffer;

	tmpBuffer = internBuffer[15];
	internBuffer[15] = internBuffer[11];
	internBuffer[11] = internBuffer[7];
	internBuffer[7] = internBuffer[3];
	internBuffer[3] = tmpBuffer;
}

__device__ unsigned char mulGaloisField2_8(unsigned char a, unsigned char b)
{
	register unsigned char p = 0;
	register unsigned char hi_bit_set;
	register unsigned char counter;

	for (counter = 0; counter < 8; counter++)
	{
		if ((b & 1) == 1)
			p ^= a;
		hi_bit_set = (a & 0x80);
		a <<= 1;
		if (hi_bit_set == 0x80)
			a ^= 0x1b;
		b >>= 1;
	}

	return p;
}

// Mix column - can be parallel
__device__ void mix_columns(unsigned char* column)
{
	register unsigned char i;
	register unsigned char cpy[4];

	#pragma unroll
	for (i = 0; i < 4; i++)
	{
		cpy[i] = column[i];
	}

	column[0] = mulGaloisField2_8(cpy[0], 2) ^
		mulGaloisField2_8(cpy[1], 3) ^
		mulGaloisField2_8(cpy[2], 1) ^
		mulGaloisField2_8(cpy[3], 1);

	column[1] = mulGaloisField2_8(cpy[0], 1) ^
		mulGaloisField2_8(cpy[1], 2) ^
		mulGaloisField2_8(cpy[2], 3) ^
		mulGaloisField2_8(cpy[3], 1);

	column[2] = mulGaloisField2_8(cpy[0], 1) ^
		mulGaloisField2_8(cpy[1], 1) ^
		mulGaloisField2_8(cpy[2], 2) ^
		mulGaloisField2_8(cpy[3], 3);

	column[3] = mulGaloisField2_8(cpy[0], 3) ^
		mulGaloisField2_8(cpy[1], 1) ^
		mulGaloisField2_8(cpy[2], 1) ^
		mulGaloisField2_8(cpy[3], 2);
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