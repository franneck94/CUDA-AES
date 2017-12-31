/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <bitset>

#include "Helper.h"
#include "Mode.h"
#include "AES.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
const vector<unsigned char*> read_datafile(const string &file_path)
{
	vector<unsigned char*> data;
	char act_char;
	unsigned int counter = 0;
	unsigned char* next_byte_array;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof())
	{
		if (counter < KEY_BLOCK)
		{
			if (counter == 0)
			{
				next_byte_array = new unsigned char[KEY_BLOCK];
			}

			infile.get(act_char);
			next_byte_array[counter] = act_char;
			counter++;
		}
		else
		{
			data.push_back(next_byte_array);
			delete next_byte_array;
			counter = 0;
		}
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

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<int> distribution(0, 15);

	for (size_t i = 0; i != length; ++i)
	{
		byte_array[i] = (unsigned char)distribution(generator);
	}

	return byte_array;
}

// Increment Counter TODO!
unsigned char* increment_counter(const unsigned char *start_counter,
								const unsigned int &round)
{
	unsigned char *test;
	test = new unsigned char[0x00, 0x00, 0x00, 0x00];
	return test;
}

// Generate Counters for all Rounds
void generate_counters(vector<unsigned char*> &ctrs, const unsigned char *IV)
{
	const unsigned int iv_size = sizeof(IV) / sizeof(unsigned char);
	unsigned char *start_counter;
	start_counter = new unsigned char[KEY_BLOCK - iv_size];
	unsigned char *ctr_i;
	ctr_i = new unsigned char[KEY_BLOCK - iv_size];
	unsigned char *res;
	res = new unsigned char[KEY_BLOCK];

	for (size_t i = 0; i != ctrs.size(); ++i)
	{
		if (i > 0)
		{
			ctr_i = increment_counter(start_counter, i);
		}

		
		for (size_t i = 0; i != KEY_BLOCK; ++i)
		{
			if (i < iv_size)
			{
				res[i] = IV[i];
			}
			else
			{
				res[i] = ctr_i[i];
			}
		}

		ctrs[i] = res;
	}
}

// Execute the Counter Mode for all Message Blocks
const vector<unsigned char*> counter_mode(const vector<unsigned char*> &messages,
										unsigned char *key,
										unsigned char *IV)
{
	AES *aes;
	unsigned char **subkeys;

	vector<unsigned char*> encrypted_messages(messages.size());
	vector<unsigned char*> ctrs(messages.size());
	generate_counters(ctrs, IV);

	for (size_t i = 0; i != messages.size(); ++i)
	{
		aes = new AES(key);
		subkeys = aes->get_subkeys();
		encrypt(ctrs[i], subkeys);
		encrypted_messages[i] = XOR(ctrs[i], messages[i]);
		delete aes;
	}

	return encrypted_messages;
}

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<unsigned char*> counter_mode_inverse(const vector<unsigned char *> &encrypted_messages,
												unsigned char *key,
												unsigned char *IV)
{
	AES *aes;
	unsigned char **subkeys;

	vector<unsigned char*> decrypted_messages(encrypted_messages.size());
	vector<unsigned char*> ctrs(encrypted_messages.size());
	generate_counters(ctrs, IV);

	for (size_t i = 0; i != encrypted_messages.size(); ++i)
	{
		aes = new AES(key);
		subkeys = aes->get_subkeys();
		encrypt(ctrs[i], subkeys);
		decrypted_messages[i] = XOR(ctrs[i], encrypted_messages[i]);
		delete aes;
	}

	return decrypted_messages;
}