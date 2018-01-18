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
	// Define Variables
	unsigned int iv_length = 12;

	for (int i = 6; i > 0; i--)
	{
		cout << endl << "Text" << i;
		string file_path_key = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/key.txt";
		string file_path_messages = "C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/text" + std::to_string(i) + ".txt";

		vector<ByteArray> decrypted_solution;
		vector<ByteArray> encrypted_solution;

		// Load data from files
		ByteArray key = read_key(file_path_key);
		ByteArray IV = random_byte_array(iv_length);
		vector<ByteArray> messages = read_datafile(file_path_messages);

		encrypted_solution = counter_mode(messages, key, IV);
		decrypted_solution = counter_mode_inverse(encrypted_solution, key, IV);

		cout << endl << "Legit solution: " << check_vector_of_byte_arrays(decrypted_solution, messages) << endl;
	}

	getchar();
}
