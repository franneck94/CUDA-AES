/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "Helper.h"
#include "AES.h"

using std::cout;
using std::endl;
using std::vector;

int main()
{
	AES aes;

	float milliseconds_encryption = 0.0f;
	float milliseconds_decryption = 0.0f;

	unsigned char *encrypted;
	unsigned char *decrypted;

	// 7649abac8119b246cee98e9b12e9197d
	unsigned char *message;
	message = new unsigned char[0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
								0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d];

	// decafbadc0deba5edeadc0debadc0ded
	unsigned char *key;
	key = new unsigned char [0xde, 0xca, 0xfb, 0xad,
							0xc0, 0xde, 0xba, 0x5e,
							0xde, 0xad, 0xc0, 0xde,
							0xba, 0xdc, 0x0d, 0xed];

	encrypted = aes.encrypt(message);

	decrypted = aes.decrypt(encrypted);

	cout << endl << "AES Algorithm runned Successfully: " << check_byte_arrays(message, decrypted) << endl;

	getchar();
}