/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <bitset>

#include "Helper.hpp"
#include "Mode.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;

#define ROUNDS 10

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Increment Counter TODO!
ByteArray increment_counter(const ByteArray &start_counter,
							const unsigned int &round)
{
	/*string next_counter_str(start_counter.begin(), start_counter.end());
	unsigned long main_hex_val = std::stol(next_counter_str);
	main_hex_val += round;
	next_counter_str = main_hex_val;*/

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
	AES aes(key);
	vector<ByteArray> encrypted_messages(messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	vector<ByteArray> ctrs(messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	generate_counters(ctrs, IV);
	int i = 0;

	// Starting Timers and Counter Mode for Encryption
	float microseconds = 0.0f;

	cout << endl << "Serial - Encrypted Duration  ";

	for (int r = 0; r != ROUNDS; ++r)
	{
		auto start_time = std::chrono::high_resolution_clock::now();

		for (i = 0; i < messages.size(); ++i)
		{
			encrypted_messages[i] = XOR(aes.encrypt(ctrs[i]), messages[i]);
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;
		microseconds += std::chrono::duration_cast<std::chrono::microseconds>(time).count();

	}

	cout << microseconds / (1000.0f * ROUNDS) << endl;
	microseconds = 0.0f;

	return encrypted_messages;
}

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<ByteArray> counter_mode_inverse(const vector<ByteArray> &encrypted_messages,
	const ByteArray &key,
	const ByteArray &IV)
{
	AES aes(key);
	vector<ByteArray> decrypted_messages(encrypted_messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	vector<ByteArray> ctrs(encrypted_messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));
	generate_counters(ctrs, IV);
	int i = 0;

	// Starting Timers and Counter Mode for Encryption
	float microseconds = 0.0f;

	cout << endl << "Serial - Decrypted Duration  ";

	for (int r = 0; r != ROUNDS; ++r)
	{
		auto start_time = std::chrono::high_resolution_clock::now();

		for (i = 0; i < encrypted_messages.size(); ++i)
		{
			decrypted_messages[i] = XOR(aes.encrypt(ctrs[i]), encrypted_messages[i]);
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;
		microseconds += std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	}

	cout << microseconds / (1000.0f * ROUNDS) << endl;
	microseconds = 0.0f;

	return decrypted_messages;
}
