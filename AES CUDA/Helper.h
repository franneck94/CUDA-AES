#pragma once

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

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
vector<unsigned char> read_datafile(const string &file_path);

// Read-In Key Datafile in Hex-Format
unsigned char* read_key(const string &file_path);

// Generate IV-Vector for Counter Mode
unsigned char* random_byte_array(const unsigned int &length);

// Cout whole unsigned char Array
void print_byte_array(unsigned char *arr, const unsigned int &size);

// Checks if two unsigned chars has same values
bool check_byte_arrays(unsigned char *arr1, unsigned char *arr2, const unsigned int &size);

// Cout hex byte
void print_byte(const unsigned char &byte);

// XOR for ByteArray
unsigned char* XOR(const unsigned char *arr1, const unsigned char *arr2);