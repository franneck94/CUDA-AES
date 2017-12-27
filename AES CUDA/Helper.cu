/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "Helper.h"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(unsigned char *arr)
{
	for (size_t i = 0; i != sizeof(arr) / sizeof(arr[0]); ++i)
	{
		cout << std::hex << (int)arr[i] << "\t";
	}
	cout << endl << endl;
}

// Checks if two ByteArrays has same values
bool check_byte_arrays(unsigned char *arr1, unsigned char *arr2)
{
	if (sizeof(arr1) != sizeof(arr2))
		return false;

	for (size_t i = 0; i != sizeof(arr1) / sizeof(arr1[0]); ++i)
	{
		if (arr1[i] != arr2[i])
			return false;
	}

	return true;
}

// Cout hex byte
void print_byte(const unsigned char &byte)
{
	cout << endl << "Byte: " << std::hex << (int)byte;
}

// Multiplication with log and exp in GF(2^8)
__device__ unsigned char mul(const unsigned char &x, const unsigned char &y, unsigned char *ltable, unsigned char *atable)
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