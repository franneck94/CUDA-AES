#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <string>
#include <iostream>

#include "Helper.h"
#include "AES.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
const vector<unsigned char*> read_datafile(const string &file_path);

// Read-In Key  Datafile in Hex-Format
unsigned char* read_key(const string &file_path);

// Generate Random ByteArray
unsigned char* random_byte_array(const unsigned int &length);

// Increment Counter TODO!
unsigned char* increment_counter(const unsigned char *start_counter,
								const unsigned int &round);

// Generate Counters for all Rounds
void generate_counters(vector<unsigned char*> &ctrs, const unsigned char *IV);

// Execute the Counter Mode for all Message Blocks
const vector<unsigned char*> counter_mode(const vector<unsigned char *> &messages,
										unsigned char *key,
										unsigned char *IV);

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<unsigned char*> counter_mode_inverse(const vector<unsigned char *> &encrypted_messages,
												unsigned char *key,
												unsigned char *IV);