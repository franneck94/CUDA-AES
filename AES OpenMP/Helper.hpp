#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

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

// Checks if two ByteArrays has same values
bool check_byte_arrays(const ByteArray &arr1, const ByteArray &arr2);

// Checks if two Vector of ByteArrays has same values
bool check_vector_of_byte_arrays(const vector<ByteArray> &arr1, const vector<ByteArray> &arr2);

// Cout hex byte
void print_byte(const unsigned char &byte);

// XOR for ByteArray
ByteArray XOR(const ByteArray &arr1, const ByteArray &arr2);