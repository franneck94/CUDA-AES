/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>

#include "Helper.hpp"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(ByteArray &arr)
{
	for (size_t i = 0; i != arr.size(); ++i)
	{
		cout << std::hex << (int)arr[i] << "\t";
	}
	cout << endl << endl;
}

// Checks if two Vector of ByteArrays has same values
bool check_vector_of_byte_arrays(const vector<ByteArray> &arr1, const vector<ByteArray> &arr2)
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
bool check_byte_arrays(const ByteArray &arr1, const ByteArray &arr2)
{
	if (arr1.size() != arr2.size())
		return false;

	for (size_t i = 0; i != arr1.size(); ++i)
	{
		if (arr1[i] != arr2[i])
		{
			cout << endl << "Error at index i2 = " << i << endl;
			return false;
		}
	}

	return true;
}

// Cout hex byte
void print_byte(const unsigned char &byte)
{
	cout << endl << "Byte: " << std::hex << (int)byte;
}

// XOR for ByteArray
ByteArray XOR(const ByteArray &arr1, const ByteArray &arr2)
{
	ByteArray res(arr1.size(), 0x00);

	for (size_t i = 0; i != arr1.size(); ++i)
	{
		res[i] = arr1[i] ^ arr2[i];
	}

	return res;
}