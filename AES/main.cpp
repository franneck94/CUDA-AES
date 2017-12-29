/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <string>

#include "Helper.hpp"
#include "AES.hpp"
#include "Mode.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

int main()
{
	// Define Variables
	unsigned int iv_length = 12;
	float milliseconds_encryption = 0.0f;
	float milliseconds_decryption = 0.0f;
	string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
	string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/decrypt.txt";
	string file_path_encrypted_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/encrypt.txt";

	// Load data from files
	ByteArray key = read_key(file_path_key);
	ByteArray IV = random_byte_array(iv_length);
	vector<ByteArray> messages = read_datafile(file_path_messages);
	vector<ByteArray> encrpyted_messages = read_datafile(file_path_encrypted_messages);

	// Starting Timers and Counter Mode for Encryption
	auto start_time = std::chrono::high_resolution_clock::now();
	counter_mode(messages, key, IV);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	milliseconds_encryption = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Serial Encrypted Duration: " << milliseconds_encryption << " (ms)." << endl;

	// Starting Timers and Counter Mode for Decryption
	start_time = std::chrono::high_resolution_clock::now();
	counter_mode_inverse(encrpyted_messages, key, IV);
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	milliseconds_encryption = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Serial Encrypted Duration: " << milliseconds_encryption << " (ms)." << endl;

	getchar();
}