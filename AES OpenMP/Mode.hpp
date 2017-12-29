#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <string>
#include <iostream>

#include "Helper.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
const vector<ByteArray> read_datafile(const string &file_path);

// Read-In Key  Datafile in Hex-Format
const ByteArray read_key(const string &file_path);

// Generate Random ByteArray
const ByteArray random_byte_array(const unsigned int &length);

// Increment Counter TODO!
ByteArray increment_counter(const ByteArray &start_counter,
	const unsigned int &round);

// Generate Counters for all Rounds
void generate_counters(vector<ByteArray> &ctrs, const ByteArray &IV);

// Execute the Counter Mode for all Message Blocks
const vector<ByteArray> counter_mode(const vector<ByteArray> &messages,
	const ByteArray &key,
	const ByteArray &IV);

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<ByteArray> counter_mode_inverse(const vector<ByteArray> &encrypted_messages,
	const ByteArray &key,
	const ByteArray &IV);