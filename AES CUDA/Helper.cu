/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <iostream>
#include <tuple>

#include "Helper.h"
#include "AES.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
std::tuple<unsigned char*, size_t> read_datafile(const string &file_path)
{
	vector<unsigned char> data;
	char act_char;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof())
	{
		infile.get(act_char);
		data.push_back((unsigned char) act_char);
	}

	infile.close();

	unsigned char *result = &data[0];

	return{ result, data.size() };
}

// Read-In Key Datafile in Hex-Format
unsigned char* read_key(const string &file_path)
{
	unsigned char *data;
	data = new unsigned char[KEY_BLOCK];
	char act_char;
	unsigned int counter = 0;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof() && counter < KEY_BLOCK)
	{
		infile.get(act_char);
		data[counter] = act_char;
		counter++;
	}

	infile.close();
	return data;
}

// Generate IV-Vector for Counter Mode
unsigned char* random_byte_array(const unsigned int &length)
{
	unsigned char *byte_array;
	byte_array = new unsigned char[length];
	size_t i = 0;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<int> distribution(0, 16);

	for (i; i != length; ++i)
	{
		byte_array[i] = (unsigned char)distribution(generator);
	}

	return byte_array;
}

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

// XOR for unsigned char*
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