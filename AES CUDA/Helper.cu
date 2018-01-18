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
vector<unsigned char> read_datafile(const string &file_path)
{
	vector<unsigned char> data;
	char act_char;
	unsigned int counter = 0;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof())
	{
		infile.get(act_char);
		data.push_back(act_char);
	}

	infile.close();
	return data;
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
void print_byte_array(unsigned char *arr, const unsigned int &size)
{
	for (size_t i = 0; i != size; ++i)
	{
		cout << std::hex << (int)arr[i] << "\t";
	}
	cout << endl << endl;
}


// Checks if two ByteArrays has same values
bool check_byte_arrays(unsigned char *arr1, unsigned char *arr2, const unsigned int &size)
{
	for (size_t i = 0; i != size; ++i)
	{
		if (arr1[i] != arr2[i])
		{ 
			cout << endl << "Error at i = " << i << " 1: " << arr1[i] << " , 2: " << arr2[i] << endl;
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