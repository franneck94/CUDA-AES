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

// Checks if two ByteArrays has same values
bool check_byte_arrays(unsigned char *arr1, unsigned char *arr2);;

// Cout hex byte
void print_byte(const unsigned char &byte);

// Multiplication with log and exp in GF(2^8)
__device__ unsigned char mul(const unsigned char &x, const unsigned char &y, unsigned char *ltable, unsigned char *atable);