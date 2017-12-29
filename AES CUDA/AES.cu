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
/*                        SUB LAYER KERNEL                           */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
__global__ void byte_sub_kernel(unsigned char *message, unsigned char *sbox)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sbox[message[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
__global__ void byte_sub_inv_kernel(unsigned char *message, unsigned char *sboxinv)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sboxinv[message[i]];
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__global__ void shift_rows_kernel(unsigned char *message)
{
	register unsigned char i = 0, j = 0, k = 0;

	for (i = 0; i != 3; ++i)
	{
		if (i == 4)
		{
			j = message[1];
			message[1] = message[5];
			message[5] = message[9];
			message[9] = message[13];
			message[13] = j;
		}
		else if (i == 8)
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
	register unsigned char i = 0, j = 0, k = 0;

	for (i = 0; i != 3; ++i)
	{
		if (i == 4)
		{
			j = message[1];
			message[1] = message[13];
			message[13] = message[9];
			message[9] = message[5];
			message[5] = j;
		}
		else if (i == 8)
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
__global__ void mix_columns_kernel(unsigned char *message, unsigned char *ltable, unsigned char *atable)
{
	register unsigned char b0, b1, b2, b3;
	register unsigned char h_02 = 0x02, h_03 = 0x03;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		b0 = message[i + 0];
		b1 = message[i + 1];
		b2 = message[i + 2];
		b3 = message[i + 3];

		// Mix-Col Matrix * b vector
		message[i + 0] = mul(b0, h_02, ltable, atable) ^ mul(b1, h_03, ltable, atable) ^ b2 ^ b3;
		message[i + 1] = b0 ^ mul(b1, h_02, ltable, atable) ^ mul(b2, h_03, ltable, atable) ^ b3;
		message[i + 2] = b0 ^ b1 ^ mul(b2, h_02, ltable, atable) ^ mul(b3, h_03, ltable, atable);
		message[i + 3] = mul(b0, h_03, ltable, atable) ^ b1 ^ b2 ^ mul(b3, h_02, ltable, atable);
	}
}

// Inverse mix column
__global__ void mix_columns_inv_kernel(unsigned char *message, unsigned char *ltable, unsigned char *atable)
{
	register unsigned char c0, c1, c2, c3;
	register unsigned char h_0e = 0x0e, h_0b = 0x0b;
	register unsigned char h_0d = 0x0d, h_09 = 0x09;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		c0 = message[i + 0];
		c1 = message[i + 1];
		c2 = message[i + 2];
		c3 = message[i + 3];

		// Mix-Col Inverse Matrix * c vector
		message[i + 0] = mul(c0, h_0e, ltable, atable) ^ mul(c1, h_0b, ltable, atable) ^ mul(c2, h_0d, ltable, atable) ^ mul(c3, h_09, ltable, atable);
		message[i + 1] = mul(c0, h_09, ltable, atable) ^ mul(c1, h_0e, ltable, atable) ^ mul(c2, h_0b, ltable, atable) ^ mul(c3, h_0d, ltable, atable);
		message[i + 2] = mul(c0, h_0d, ltable, atable) ^ mul(c1, h_09, ltable, atable) ^ mul(c2, h_0e, ltable, atable) ^ mul(c3, h_0b, ltable, atable);
		message[i + 3] = mul(c0, h_0b, ltable, atable) ^ mul(c1, h_0d, ltable, atable) ^ mul(c2, h_09, ltable, atable) ^ mul(c3, h_0e, ltable, atable);
	}
}

// Key Addition Kernel
__global__ void key_addition_kernel(unsigned char *message, unsigned char *subkey)
{
	register int i = 0;
	register unsigned int size = sizeof(message) / sizeof(unsigned char);

	for (i; i != size; i++)
		message[i] ^= subkey[i];
}

/*********************************************************************/
/*                           CONSTRUCTORS                            */
/*********************************************************************/

// Constructor of AES en/decryption
AES::AES(unsigned char *message, unsigned char *key) : m_subkeys(SUB_KEYS)
{
	m_message = message;
	m_key = key;
	key_schedule();
}

/*********************************************************************/
/*                       EN- DECRYPTION FUNCTIONS                    */
/*********************************************************************/

// Starting the encryption phase
unsigned char* AES::encrypt()
{
	register int round = 0;
	unsigned char *message = m_message;

	// Key-Add before round 1 (R0)
	key_addition(message, round);
	round = 1;

	// Round 1 to NUM_ROUNDS - 1 (R1 to R9)
	for (round; round != NUM_ROUNDS; round++)
	{
		byte_sub(message);
		shift_rows(message);
		mix_columns(message);
		key_addition(message, round);
	}

	// Last round without Mix-Column (RNUM_ROUNDS)
	round = NUM_ROUNDS;
	byte_sub(message);
	shift_rows(message);
	key_addition(message, round);

	return message;
}

// Starting the decryption phase
unsigned char* AES::decrypt()
{
	register int round = NUM_ROUNDS;
	unsigned char *message = m_message;

	// Key-Add before round (Inverse NUM_ROUNDS)
	key_addition(message, round);
	shift_rows_inv(message);
	byte_sub_inv(message);
	round = NUM_ROUNDS - 1;

	// Round NUM_ROUNDS - 1 to 1 (Inverse R9 to R1)
	for (round; round > 0; round--)
	{
		key_addition(message, round);
		mix_columns_inv(message);
		shift_rows_inv(message);
		byte_sub_inv(message);
	}

	// Last round without Mix-Column (Inverse R0)
	round = 0;
	key_addition(message, round);

	return message;
}

/*********************************************************************/
/*                           KEY FUNCTIONS                           */
/*********************************************************************/

// Computing the round keys
void AES::key_schedule()
{
	register int r;

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

	register int i;

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

/*********************************************************************/
/*                              SUB LAYER                            */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
void AES::byte_sub(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_sbox = sizeof(sbox);
	unsigned char *d_sbox;
	cudaMalloc((void **)&d_sbox, size_sbox);
	cudaMemcpy(d_sbox, sbox, size_sbox, cudaMemcpyHostToDevice);

	byte_sub_kernel <<<1, 1>>>(d_message, d_sbox);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_sbox);
}

// Inverse byte substitution (S-Boxes) can be parallel
void AES::byte_sub_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_sboxinv = sizeof(sboxinv);
	unsigned char *d_sboxinv;
	cudaMalloc((void **)&d_sboxinv, size_sboxinv);
	cudaMemcpy(d_sboxinv, sboxinv, size_sboxinv, cudaMemcpyHostToDevice);

	byte_sub_inv_kernel <<<1, 1 >>>(d_message, d_sboxinv);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_sboxinv);
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
void AES::shift_rows(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	shift_rows_kernel <<<1, 1 >>>(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
void AES::shift_rows_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	shift_rows_inv_kernel<<<1, 1 >>>(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Mix column - can be parallel
void AES::mix_columns(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_ltable = sizeof(ltable);
	unsigned char *d_ltable;
	cudaMalloc((void **)&d_ltable, size_ltable);
	cudaMemcpy(d_ltable, ltable, size_ltable, cudaMemcpyHostToDevice);

	int size_atable = sizeof(atable);
	unsigned char *d_atable;
	cudaMalloc((void **)&d_atable, size_atable);
	cudaMemcpy(d_atable, atable, size_atable, cudaMemcpyHostToDevice);

	mix_columns_kernel <<<1, 1 >>>(d_message, d_ltable, d_atable);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Inverse mix column
void AES::mix_columns_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_ltable = sizeof(ltable);
	unsigned char *d_ltable;
	cudaMalloc((void **)&d_ltable, size_ltable);
	cudaMemcpy(d_ltable, ltable, size_ltable, cudaMemcpyHostToDevice);

	int size_atable = sizeof(atable);
	unsigned char *d_atable;
	cudaMalloc((void **)&d_atable, size_atable);
	cudaMemcpy(d_atable, atable, size_atable, cudaMemcpyHostToDevice);

	mix_columns_inv_kernel <<<1, 1 >>>(d_message, d_ltable, d_atable);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

void AES::key_addition(unsigned char *message, const int &r)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_subkey = sizeof(m_subkeys[r]);
	unsigned char *d_subkey = &m_subkeys[r][0];
	cudaMalloc((void **)&d_subkey, size_subkey);
	cudaMemcpy(d_subkey, &m_subkeys[r], size_subkey, cudaMemcpyHostToDevice);

	key_addition_kernel <<<1, 1 >>>(d_message, d_subkey);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_subkey);
}