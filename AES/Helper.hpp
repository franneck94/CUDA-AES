/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#pragma once
#include <vector>
#include <iostream>
#include <bitset>

#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(ByteArray &arr);

// Cout hex byte
void print_byte(const unsigned char &byte);

// Multiplication with shift and mod in GF(2^8)
inline unsigned char mul_shift(const unsigned char &x, const unsigned char &y)
{
	unsigned char result;

	if (y == 0x02)
	{
		result = (x << 1);
		if (x & 0x80);
		{
			result ^= 0x1b;
		}
		return result;
	}
	else if (y == 0x03)
	{
		result = (x << 1);
		if (x & 0x80);
		{
			result ^= 0x1b;
		}
		result ^= x;
		return result;
	}
}

// Multiplication with log and exp in GF(2^8)
inline unsigned char mul(const unsigned char &x, const unsigned char &y)
{
	int s;
	int q;
	int z = 0;

	s = ltable[x] + ltable[y];
	s %= 255;
	s = atable[s];
	q = s;

	if (x == 0)
		s = z;
	else 
		s = q;

	if (y == 0) 
		s = z;
	else 
		q = z;

	return s;
}