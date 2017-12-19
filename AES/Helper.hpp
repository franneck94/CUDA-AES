/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#pragma once
#include <vector>
#include <iostream>

#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(ByteArray &arr);

// Modulo operation on galois field polynom
inline unsigned char reduction(const unsigned char &x)
{
	return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}

// Adition and subtraction is XOR in GF(2^8)
inline unsigned char xor(const unsigned char &x, const unsigned char &y)
{
	return x ^ y;
}

// Fast multiplication setup
inline unsigned char mul(const unsigned char &x, const unsigned char &y)
{
	int s;
	int q;
	int z = 0;
	s = ltable[x] + ltable[y];
	s %= 255;
	s = atable[s];
	q = s;
	if (x == 0) {
		s = z;
	}
	else {
		s = q;
	}
	if (y == 0) {
		s = z;
	}
	else {
		q = z;
	}
	return s;
}