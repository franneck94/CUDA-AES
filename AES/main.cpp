/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>

#include "Helper.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;

int main()
{
	AES aes;

	float milliseconds_encryption = 0.0f;
	float milliseconds_decryption = 0.0f;

	ByteArray encrypted;
	ByteArray decrypted;

	// 7649abac8119b246cee98e9b12e9197d
	ByteArray message = { 0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
		0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d };

	// decafbadc0deba5edeadc0debadc0ded
	ByteArray key = { 0xde, 0xca, 0xfb, 0xad,
		0xc0, 0xde, 0xba, 0x5e,
		0xde, 0xad, 0xc0, 0xde,
		0xba, 0xdc, 0x0d, 0xed };

	auto start_time = std::chrono::high_resolution_clock::now();
	encrypted = aes.encrypt(message);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	milliseconds_encryption = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Encrypted Duration: " << milliseconds_encryption << " (ms)." << endl;

	start_time = std::chrono::high_resolution_clock::now();
	decrypted = aes.decrypt(encrypted);
	end_time = std::chrono::high_resolution_clock::now();
	time = end_time - start_time;
	milliseconds_decryption = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
	cout << endl << "Decrypted Duration: " << milliseconds_decryption << " (ms)." << endl;

	cout << endl << "AES Algorithm runned Successfully: " << check_byte_arrays(message, decrypted) << endl;

	getchar();
}