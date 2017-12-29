/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <string>
#include <iostream>

#include "Helper.hpp"
#include "Mode.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

/*********************************************************************/
/*                     COUNTER MODE FUNCTIONS                        */
/*********************************************************************/

// Read-In Datafile in Hex-Format and Vector of ByteArrays
const vector<ByteArray> read_datafile(const string &file_path)
{
	vector<ByteArray> data;

	return data;
}

// Read-In Key Datafile in Hex-Format
const ByteArray read_key(const string &file_path)
{
	ByteArray data;

	return data;
}

// Generate IV-Vector for Counter Mode
const ByteArray generate_iv(const unsigned int &salt)
{
	return ByteArray(0x00);
}

// Execute the Counter Mode for all Message Blocks
const vector<ByteArray> counter_mode(const vector<ByteArray> &messages, 
									const ByteArray &key,
									const ByteArray &IV)
{
	vector<ByteArray> encrypted_messages(messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));

	for (size_t i = 0; i != messages.size(); ++i)
	{
		if (i == 0)
		{
			// TODO
		}
		// TODO
	}

	return encrypted_messages;
}

// Execute the Inverse Counter Mode for all Decrypted Message Blocks
const vector<ByteArray> counter_mode_inverse(const vector<ByteArray> &encrypted_messages,
											const ByteArray &key,
											const ByteArray &IV)
{
	vector<ByteArray> decrypted_messages(encrypted_messages.size(), vector<unsigned char>(KEY_BLOCK, 0x00));

	for (size_t i = 0; i != encrypted_messages.size(); ++i)
	{
		if (i == 0)
		{
			// TODO
		}
		// TODO
	}

	return decrypted_messages;
}