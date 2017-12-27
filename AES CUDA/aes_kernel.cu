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
#include "Table.h"

/*********************************************************************/
/*                        SUB LAYER KERNEL                           */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
__device__ void byte_sub_kernel(unsigned char *message, const unsigned char *sbox)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sbox[message[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
__device__ void byte_sub_inv_kernel(unsigned char *message, const unsigned char *sboxinv)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; i++)
		message[i] = sboxinv[message[i]];
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
__device__ void shift_rows_kernel(unsigned char *message)
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
__device__ void shift_rows_inv_kernel(unsigned char *message)
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
__device__ void mix_columns_kernel(unsigned char *message, unsigned char *ltable, unsigned char *atable)
{
	register unsigned char b0, b1, b2, b3;
	const register unsigned char h_02 = 0x02, h_03 = 0x03;
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
__device__ void mix_columns_inv_kernel(unsigned char *message, unsigned char *ltable, unsigned char *atable)
{
	register unsigned char c0, c1, c2, c3;
	const register unsigned char h_0e = h_0e, h_0b = h_0b;
	const register unsigned char h_0d = h_0d, h_09 = h_09;
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

__device__ void key_addition_kernel(unsigned char *message, unsigned char *subkey, const int &size)
{
	register int i = 0;

	for (i; i != size; i++)
		message[i] ^= subkey[i];
}