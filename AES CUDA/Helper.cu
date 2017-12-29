/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <iostream>

#include "Helper.h"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole unsigned char Array
void print_byte_array(unsigned char *arr)
{
	for (size_t i = 0; i != sizeof(arr) / sizeof(arr[0]); ++i)
	{
		cout << std::hex << (int)arr[i] << "\t";
	}
	cout << endl << endl;
}

// Checks if two Vector of unsigned chars has same values
bool check_vector_of_byte_arrays(const vector<unsigned char*> &arr1, const vector<unsigned char*> &arr2)
{
	bool check = true;

	if (arr1.size() != arr2.size())
		return false;

	for (size_t i = 0; i != arr1.size(); ++i)
	{
		if (arr1[i] != arr2[i])
			check = check_byte_arrays(arr1[i], arr2[i]);
		if (!check)
		{
			cout << endl << "Error at index i = " << i << endl;
			return false;
		}
	}

	return true;
}

// Checks if two ByteArrays has same values
bool check_byte_arrays(const unsigned char *arr1, const unsigned char *arr2)
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
__device__ unsigned char mul(unsigned char &x, unsigned char &y, unsigned char *ltable, unsigned char *atable)
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

// XOR for ByteArray
unsigned char* XOR(const unsigned char *arr1, const unsigned char *arr2)
{
	unsigned char* res;
	const unsigned int arr_size = sizeof(arr1) / sizeof(unsigned char);
	res = new unsigned char[arr_size];

	for (size_t i = 0; i != arr_size; ++i)
	{
		res[i] = arr1[i] ^ arr2[i];
	}

	return res;
}