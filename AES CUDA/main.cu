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

	// Load data from files
	const unsigned char *key = read_key(file_path_key);
	const unsigned char *IV = random_byte_array(iv_length);
	vector<unsigned char*> messages = read_datafile(file_path_messages);

	cout << endl << "Starting AES CUDA - COUNTER MODE, with Key: " << endl;
	print_byte_array(key);

	// Starting Timers and Counter Mode for Encryption
	auto start_time = std::chrono::high_resolution_clock::now();
	// COUNTER
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "CUDA Encrypted Duration: " << microseconds << " (us)." << endl;

	// Starting Timers and Counter Mode for Decryption
	start_time = std::chrono::high_resolution_clock::now();
	// INVERSE COUNTER
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "CUDA Encrypted Duration: " << microseconds << " (us)." << endl;

	cout << endl << "Legit solution: " << check_vector_of_byte_arrays(decrypted_solution, messages) << endl;

	getchar();
}