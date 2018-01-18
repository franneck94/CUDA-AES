/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>

#include "Helper.hpp"
#include "AES.hpp"
#include "Mode.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

int main()
{
  //    omp_set_nested(1);
	// Define Variables
	unsigned int iv_length = 12;
	float microseconds = 0.0f;

	string file_path_key = "./key";
	string file_path_messages = "./data";

	vector<ByteArray> decrypted_solution;
	vector<ByteArray> encrypted_solution;

	cout << endl << "Starting!";

	// Load data from files
	ByteArray key = read_key(file_path_key);
	ByteArray IV = random_byte_array(iv_length);
	vector<ByteArray> messages = read_datafile(file_path_messages);

	// Starting Timers and Counter Mode for Encryption
	auto start_time = std::chrono::high_resolution_clock::now();
	encrypted_solution = counter_mode(messages, key, IV);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "OpenMP Encrypted Duration: " << microseconds << " (us)." << endl;

	// Starting Timers and Counter Mode for Decryption
	start_time = std::chrono::high_resolution_clock::now();
	decrypted_solution = counter_mode_inverse(encrypted_solution, key, IV);
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "OpenMP Encrypted Duration: " << microseconds << " (us)." << endl;

	cout << endl << "Legit solution: " << check_vector_of_byte_arrays(decrypted_solution, messages) << endl;

	//first message
	int k = 0;
	cout << "Decrypted:" << endl;
	for (std::vector<unsigned char>::const_iterator i = decrypted_solution[k].begin(); i != decrypted_solution[k].end(); ++i)
	  cout << *i << ' ';

	  cout << endl;

	  cout << "Encrypted:" << endl;
      	  for (std::vector<unsigned char>::const_iterator i = encrypted_solution[k].begin(); i != encrypted_solution[k].end(); ++i)
	  cout << *i << ' ';

	  cout << endl;

	  cout << "Original:" << endl;
	for (std::vector<unsigned char>::const_iterator i = messages[k].begin(); i != messages[k].end(); ++i)
	  cout << *i << ' ';

	cout << endl;
	//second message
	k = 1;
	cout << "Decrypted:" << endl;
	for (std::vector<unsigned char>::const_iterator i = decrypted_solution[k].begin(); i != decrypted_solution[k].end(); ++i)
	  cout << *i << ' ';

	  cout << endl;

	  cout << "Encrypted:" << endl;
	  for (std::vector<unsigned char>::const_iterator i = encrypted_solution[k].begin(); i != encrypted_solution[k].end(); ++i)
	    cout << *i << ' ';

	  cout << endl;

	  cout << "Original:" << endl;
	for (std::vector<unsigned char>::const_iterator i = messages[k].begin(); i != messages[k].end(); ++i)
	  cout << *i << ' ';

	cout << endl;
	//third message
	k = 2;
	cout << "Decrypted:" << endl;
	for (std::vector<unsigned char>::const_iterator i = decrypted_solution[k].begin(); i != decrypted_solution[k].end(); ++i)
	  cout << *i << ' ';

	  cout << endl;

	  cout << "Encrypted:" << endl;
	  for (std::vector<unsigned char>::const_iterator i = encrypted_solution[k].begin(); i != encrypted_solution[k].end(); ++i)
	    cout << *i << ' ';

	  cout << endl;

	  cout << "Original:" << endl;
	for (std::vector<unsigned char>::const_iterator i = messages[k].begin(); i != messages[k].end(); ++i)
	  cout << *i << ' ';

	cout << endl << "decrypted_solution[k].size() = " << decrypted_solution[k].size() << endl;
	cout << "encrypted_solution[k].size() = " << encrypted_solution[k].size() << endl;
	cout << "messages[k].size() = " << messages[k].size() << endl;

	getchar();
}
