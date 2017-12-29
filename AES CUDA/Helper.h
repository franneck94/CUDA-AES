#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

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

// Cout whole unsigned char Array
void print_byte_array(unsigned char *arr);

// Checks if two unsigned chars has same values
bool check_byte_arrays(const unsigned char *arr1, const unsigned char *arr2);

// Checks if two Vector of unsigned chars has same values
bool check_vector_of_byte_arrays(const vector<unsigned char*> &arr1, const vector<unsigned char*> &arr2);

// Cout hex byte
void print_byte(const unsigned char &byte);

// Multiplication with log and exp in GF(2^8)
__device__ unsigned char mul(const unsigned char &x, const unsigned char &y, unsigned char *ltable, unsigned char *atable);

// XOR for ByteArray
unsigned char* XOR(const unsigned char *arr1, const unsigned char *arr2);