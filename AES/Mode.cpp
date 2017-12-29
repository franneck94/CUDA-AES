/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <bitset>

#include "Helper.hpp"
#include "Mode.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
const vector<ByteArray> read_datafile(const string &file_path)
{
	vector<ByteArray> data;
	char act_char;
	unsigned int counter = 0;
	ByteArray next_byte_array;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof())
	{
		if (counter < KEY_BLOCK)
		{
			infile.get(act_char);
			next_byte_array.push_back(act_char);
			counter++;
		}
		else
		{
			data.push_back(next_byte_array);
			next_byte_array = {};
			counter = 0;
		}
	}

	infile.close();
	return data;
}

// Read-In Key Datafile in Hex-Format
const ByteArray read_key(const string &file_path)
{
	ByteArray data;
	char act_char;
	unsigned int counter = 0;
	ifstream infile;

	infile.open(file_path);

	while (!infile.eof() && counter < KEY_BLOCK)
	{
		infile.get(act_char);
		data.push_back(act_char);
		counter++;
	}

	infile.close();
	return data;
}

// Generate IV-Vector for Counter Mode
const ByteArray random_byte_array(const unsigned int &length)
{
	ByteArray byte_array(length);

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<int> distribution(0, 16);

	for (size_t i = 0; i != byte_array.size(); ++i)
	{
		byte_array[i] = (unsigned char) distribution(generator);
	}

	return byte_array;
}

// Increment Counter TODO!
ByteArray increment_counter(const ByteArray &start_counter,
							const unsigned int &round)
{
	//string next_counter_str(start_counter.begin(), start_counter.end());
	//long test = (long)next_counter_str.c_str();
	//test += round;
	//next_counter_str = test;
	//ByteArray next_counter(next_counter_str.begin(), next_counter_str.end());

	//return next_counter;
	ByteArray test{ 0x00, 0x00, 0x00, 0x00 };
	return test;
}

// Generate Counters for all Rounds
void generate_counters(vector<ByteArray> &ctrs, const ByteArray &IV)
{
	ByteArray start_counter(KEY_BLOCK - IV.size(), 0x00);
	ByteArray ctr_i(KEY_BLOCK - IV.size(), 0x00);
	ByteArray res(KEY_BLOCK, 0x00);

	for (size_t i = 0; i != ctrs.size(); ++i)
	{
		res = IV;

		if (i > 0)
		{
			ctr_i = increment_counter(start_counter, i);
		}

		res.insert(res.end(), ctr_i.begin(), ctr_i.end());
		ctrs[i] = res;
	}
}

// Execute the Counter Mode for all Message Blocks
const vector<ByteArray> counter_mode(const vector<ByteArray> &messages, 
									const ByteArray &key,
									const ByteArray &IV)
{
	AES *aes;
	vector<ByteArray> encrypted_messages(messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	vector<ByteArray> ctrs(messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	generate_counters(ctrs, IV);

	for (size_t i = 0; i != messages.size(); ++i)
	{
		aes = new AES(ctrs[i], key);
		encrypted_messages[i] = XOR(aes->encrypt(), messages[i]);
		delete aes;
	}

	return encrypted_messages;
}

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<ByteArray> counter_mode_inverse(const vector<ByteArray> &encrypted_messages,
											const ByteArray &key,
											const ByteArray &IV)
{
	AES *aes;
	vector<ByteArray> decrypted_messages(encrypted_messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	vector<ByteArray> ctrs(encrypted_messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	generate_counters(ctrs, IV);

	for (size_t i = 0; i != encrypted_messages.size(); ++i)
	{
		aes = new AES(ctrs[i], key);
		decrypted_messages[i] = XOR(aes->decrypt(), encrypted_messages[i]);
		delete aes;
	}

	return decrypted_messages;
}