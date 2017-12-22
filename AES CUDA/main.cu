/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>

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

	cout << endl << "Message: " << endl;
	print_byte_array(message);
	cout << endl << "Key: " << endl;
	print_byte_array(key);

	cout << endl << "Encrypted: " << endl;
	encrypted = aes.encrypt(message);
	print_byte_array(encrypted);
	cout << endl << "Decrypted: " << endl;
	decrypted = aes.decrypt(encrypted);
	print_byte_array(decrypted);

	getchar();
}