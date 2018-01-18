/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

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

__device__ static const unsigned char iv[KEY_BLOCK] = { 0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00 };

/*********************************************************************/
/*                           MAIN KERNEL                             */
/*********************************************************************/

__global__ void cuda_aes_encrypt_ctr(unsigned char *in, unsigned char *out, int n, int counter)
{
	//Map the thread id and block id to the AES block
	int i = ((blockIdx.x * THREADS_PER_BLOCK) + threadIdx.x) * KEY_BLOCK;

	if (i < n)
	{
		//Call the encrypt function on the current 16-unsigned char block
		aes_encrypt_ctr(&in[i], &out[i], d_keySchedule, AES_BITS, counter);
	}
}

__global__ void cuda_aes_encrypt(unsigned char *in, unsigned char *out, int n)
{
	//Map the thread id and block id to the AES block
	int i = ((blockIdx.x * THREADS_PER_BLOCK) + threadIdx.x) * KEY_BLOCK;

	if (i < n)
	{
		//Call the encrypt function on the current 16-unsigned char block
		aes_encrypt(&in[i], &out[i], d_keySchedule, AES_BITS);
	}
}

__global__ void cuda_aes_decrypt(unsigned char *in, unsigned char *out, int n)
{
	//Map the thread id and block id to the AES block
	int i = ((blockIdx.x * THREADS_PER_BLOCK) + threadIdx.x) * KEY_BLOCK;

	if (i < n)
	{
		//Call the encrypt function on the current 16-unsigned char block
		aes_decrypt(&in[i], &out[i], d_keySchedule, AES_BITS);
	}
}

/*********************************************************************/
/*                         MAIN DEVICE KERNEL                        */
/*********************************************************************/

__device__ void aes_encrypt_ctr(const unsigned char in[], unsigned char out[], 
							const unsigned char key[], int keysize, int counter)
{

}

__device__ void aes_encrypt(const unsigned char in[], unsigned char out[], 
							const unsigned char key[], int keysize)
{

}

__device__ void aes_decrypt(const unsigned char in[], unsigned char out[], 
							const unsigned char key[], int keysize)
{

}

/*********************************************************************/
/*                      SUB LAYER DEVICE KERNEL                      */
/*********************************************************************/

//// unsigned char substitution (S-Boxes) can be parallel
//__global__ void byte_sub_kernel(unsigned char *message)
//{
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (id < KEY_BLOCK)
//	{
//		message[id] = d_sbox[message[id]];
//	}
//}
//
//// Inverse unsigned char substitution (S-Boxes) can be parallel
//__global__ void byte_sub_inv_kernel(unsigned char *message)
//{
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (id < KEY_BLOCK)
//	{
//		message[id] = d_sboxinv[message[id]];
//	}
//}
//
//// Shift rows - can be parallel
//// B0, B4, B8, B12 stays the same
//__global__ void shift_rows_kernel(unsigned char *message)
//{
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//	unsigned char j = 0, k = 0;
//
//	if (id < SHIFT_ROW_LIMIT)
//	{
//		if (id == 0)
//		{
//			j = message[1];
//			message[1] = message[5];
//			message[5] = message[9];
//			message[9] = message[13];
//			message[13] = j;
//		}
//		else if (id == 1)
//		{
//			j = message[10];
//			k = message[14];
//			message[10] = message[2];
//			message[2] = j;
//			message[14] = message[6];
//			message[6] = k;
//		}
//		else
//		{
//			k = message[3];
//			message[3] = message[15];
//			message[15] = message[11];
//			message[11] = message[7];
//			message[7] = k;
//		}
//	}
//}
//
//// Inverse shift rows - can be parallel
//// C0, C4, C8, C12 stays the same
//__global__ void shift_rows_inv_kernel(unsigned char *message)
//{
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//	unsigned char j = 0, k = 0;
//
//	if (id < SHIFT_ROW_LIMIT)
//	{
//		if (id == 0)
//		{
//			j = message[1];
//			message[1] = message[13];
//			message[13] = message[9];
//			message[9] = message[5];
//			message[5] = j;
//		}
//		else if (id == 1)
//		{
//			j = message[2];
//			k = message[6];
//			message[2] = message[10];
//			message[10] = j;
//			message[6] = message[14];
//			message[14] = k;
//		}
//		else
//		{
//			j = message[3];
//			message[3] = message[7];
//			message[7] = message[11];
//			message[11] = message[15];
//			message[15] = j;
//		}
//	}
//}
//
//// Mix column - can be parallel
//__global__ void mix_columns_kernel(unsigned char *message)
//{
//	unsigned char b0, b1, b2, b3;
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (id < MIX_COLUMN_LIMIT)
//	{
//		b0 = message[id + 0];
//		b1 = message[id + 1];
//		b2 = message[id + 2];
//		b3 = message[id + 3];
//
//		// Mix-Col Matrix * b vector
//		message[id + 0] = d_mul[b0][0] ^ d_mul[b1][1] ^ b2 ^ b3;
//		message[id + 1] = b0 ^ d_mul[b1][0] ^ d_mul[b2][1] ^ b3;
//		message[id + 2] = b0 ^ b1 ^ d_mul[b2][0] ^ d_mul[b3][1];
//		message[id + 3] = d_mul[b0][1] ^ b1 ^ b2 ^ d_mul[b3][0];
//	}
//}
//
//// Inverse mix column
//__global__ void mix_columns_inv_kernel(unsigned char *message)
//{
//	unsigned char c0, c1, c2, c3;
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (id < MIX_COLUMN_LIMIT)
//	{
//		c0 = message[id + 0];
//		c1 = message[id + 1];
//		c2 = message[id + 2];
//		c3 = message[id + 3];
//
//		// Mix-Col Inverse Matrix * c vector
//		message[id + 0] = d_mul[c0][5] ^ d_mul[c1][3] ^ d_mul[c2][4] ^ d_mul[c3][2];
//		message[id + 1] = d_mul[c0][2] ^ d_mul[c1][5] ^ d_mul[c2][3] ^ d_mul[c3][4];
//		message[id + 2] = d_mul[c0][4] ^ d_mul[c1][2] ^ d_mul[c2][5] ^ d_mul[c3][3];
//		message[id + 3] = d_mul[c0][3] ^ d_mul[c1][4] ^ d_mul[c2][2] ^ d_mul[c3][5];
//	}
//}
//
//// Key Addition Kernel
//__global__ void key_addition_kernel(unsigned char *message, unsigned char **subkeys, const unsigned int &round)
//{
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (id < KEY_BLOCK)
//	{
//		message[id] = message[id] ^ subkeys[round][id];
//	}
//}