/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <string>

#include "Helper.h"
#include "AES.h"
#include "Mode.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

int main()
{
	// Define Variables
	unsigned int iv_length = 12;
	float microseconds = 0.0f;

	string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
	string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/text.txt";;

	vector<unsigned char*> decrypted_solution;
	vector<unsigned char*> encrypted_solution;

	//// 7649abac8119b246cee98e9b12e9197d
	//unsigned char message[KEY_BLOCK] = { 0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
	//							0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d };
	//unsigned char message_check[KEY_BLOCK] = { 0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
	//	0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d };

	//// decafbadc0deba5edeadc0debadc0ded
	//unsigned char key[KEY_BLOCK] = { 0xde, 0xca, 0xfb, 0xad,
	//						0xc0, 0xde, 0xba, 0x5e,
	//						0xde, 0xad, 0xc0, 0xde,
	//						0xba, 0xdc, 0x0d, 0xed };


	/*AES aes(key);
	unsigned char** subkeys = aes.get_subkeys();*/

	/*print_byte_array(message);
	auto start_time = std::chrono::high_resolution_clock::now();
	encrypt(message, subkeys);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Encrypted Duration: " << microseconds << " (us)." << endl;

	print_byte_array(message);
	start_time = std::chrono::high_resolution_clock::now();
	decrypt(message, subkeys);
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Encrypted Duration: " << microseconds << " (us)." << endl;
	print_byte_array(message);

	cout << endl << "AES Algorithm runned Successfully: " << check_byte_arrays(message, message_check) << endl;*/

	// Load data from files
	unsigned char *key = read_key(file_path_key);
	unsigned char *IV = random_byte_array(iv_length);
	vector<unsigned char*> messages = read_datafile(file_path_messages);

	// Starting Timers and Counter Mode for Encryption
	auto start_time = std::chrono::high_resolution_clock::now();
	encrypted_solution = counter_mode(messages, key, IV);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "CUDA Encrypted Duration: " << microseconds << " (us)." << endl;

	// Starting Timers and Counter Mode for Decryption
	start_time = std::chrono::high_resolution_clock::now();
	decrypted_solution = counter_mode_inverse(encrypted_solution, key, IV);
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "CUDA Encrypted Duration: " << microseconds << " (us)." << endl;

	cout << endl << "Legit solution: " << check_vector_of_byte_arrays(decrypted_solution, messages) << endl;

	getchar();
}