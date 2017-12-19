/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "Helper.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;

int main()
{
	AES aes;

	// Fallsch
	//ByteArray buffer = { 0x98, 0x2b, 0x0c, 0xbf, 
	//					0xdd, 0x97, 0xf8, 0x99,
	//					0x9c, 0x3a, 0x34, 0x0b, 
	//					0xaa, 0x56, 0xcd, 0x19};

	// Richtig
	/*ByteArray buffer = { 0x98, 0x21, 0x0c, 0xbf,
						0xdd, 0x9d, 0xf8, 0x99,
						0x9c, 0x30, 0x34, 0x0b,
						0xaa, 0x5c, 0xcd, 0x19 };
	aes.mix_columns_inv(buffer);

	print_byte_array(buffer);*/

	ByteArray encrypted;
	ByteArray decrypted;
	ByteArray message = { 0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
						0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d };

	encrypted = aes.encrypt(message);
	decrypted = aes.decrypt(encrypted);

	cout << endl << "Encrypted: " << endl;
	print_byte_array(encrypted);
	cout << endl << "Decrypted: " << endl;
	print_byte_array(decrypted);
	cout << endl << "Message: " << endl;
	print_byte_array(message);

	getchar();
}