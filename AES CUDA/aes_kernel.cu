/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "AES.h"
#include "Helper.h"

/*********************************************************************/
/*                        SUB LAYER KERNEL                           */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
__device__ void byte_sub_kernel(unsigned char *message)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sbox[message[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
__device__ void byte_sub_inv_kernel(unsigned char *message)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sboxinv[message[i]];
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__device__ void shift_rows_kernel(unsigned char *message)
{
	register unsigned char i, j, k, l;

	i = message[1];
	message[1] = message[5];
	message[5] = message[9];
	message[9] = message[13];
	message[13] = i;

	j = message[10];
	message[10] = message[2];
	message[2] = j;
	l = message[14];
	message[14] = message[6];
	message[6] = l;

	k = message[3];
	message[3] = message[15];
	message[15] = message[11];
	message[11] = message[7];
	message[7] = k;
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
__device__ void shift_rows_inv_kernel(unsigned char *message)
{
	register unsigned char i, j, k, l;

	i = message[1];
	message[1] = message[13];
	message[13] = message[9];
	message[9] = message[5];
	message[5] = i;

	j = message[2];
	message[2] = message[10];
	message[10] = j;
	l = message[6];
	message[6] = message[14];
	message[14] = l;

	k = message[3];
	message[3] = message[7];
	message[7] = message[11];
	message[11] = message[15];
	message[15] = k;
}

// Mix column - can be parallel
__device__ void mix_columns_kernel(unsigned char *message)
{
	register unsigned char b0, b1, b2, b3;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		b0 = message[i + 0];
		b1 = message[i + 1];
		b2 = message[i + 2];
		b3 = message[i + 3];

		// Mix-Col Matrix * b vector
		message[i + 0] = mul(b0, 0x02) ^ mul(b1, 0x03) ^ b2 ^ b3;
		message[i + 1] = b0 ^ mul(b1, 0x02) ^ mul(b2, 0x03) ^ b3;
		message[i + 2] = b0 ^ b1 ^ mul(b2, 0x02) ^ mul(b3, 0x03);
		message[i + 3] = mul(b0, 0x03) ^ b1 ^ b2 ^ mul(b3, 0x02);
	}
}

// Inverse mix column
__device__ void mix_columns_inv_kernel(unsigned char *message)
{
	register unsigned char c0, c1, c2, c3;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		c0 = message[i + 0];
		c1 = message[i + 1];
		c2 = message[i + 2];
		c3 = message[i + 3];

		// Mix-Col Inverse Matrix * c vector
		message[i + 0] = mul(c0, 0x0e) ^ mul(c1, 0x0b) ^ mul(c2, 0x0d) ^ mul(c3, 0x09);
		message[i + 1] = mul(c0, 0x09) ^ mul(c1, 0x0e) ^ mul(c2, 0x0b) ^ mul(c3, 0x0d);
		message[i + 2] = mul(c0, 0x0d) ^ mul(c1, 0x09) ^ mul(c2, 0x0e) ^ mul(c3, 0x0b);
		message[i + 3] = mul(c0, 0x0b) ^ mul(c1, 0x0d) ^ mul(c2, 0x09) ^ mul(c3, 0x0e);
	}
}

__device__ void key_addition_kernel(unsigned char *message, unsigned char *subkey, const int &size)
{
	register int i = 0;

	for (i; i != size; i++)
		message[i] ^= subkey[i];
}