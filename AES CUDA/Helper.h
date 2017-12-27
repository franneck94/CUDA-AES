#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>
#include <bitset>

#include "AES.h"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(unsigned char *arr);

// Cout hex byte
void print_byte(const unsigned char &byte);

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