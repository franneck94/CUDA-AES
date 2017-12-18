/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "AES.hpp"

#include <iostream>
#include <stdlib.h>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

unsigned char reduction(unsigned char x);

/*********************************************************************/
/*                           CONSTRUCTORS                            */
/*********************************************************************/

// Constructor of AES en/decryption
AES::AES(const ByteArray &key)
{
	cout << "AES datastream created!" << endl;
}

// Destructor of AES en/decryption
AES::~AES()
{
	cout << "AES done!" << endl;
}


/*********************************************************************/
/*                       EN- DECRYPTION FUNCTIONS                    */
/*********************************************************************/

// Starting the encryption phase
void AES::encrypt(unsigned char *buffer)
{

}

// Starting the decryption phase
void AES::decrypt(unsigned char *buffer)
{

}

// Computing the round keys
void AES::key_schedule(const ByteArray &key)
{

}

/*********************************************************************/
/*                              SUB LAYER                            */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
void AES::byte_sub(unsigned char* buffer)
{
	register unsigned char i = KEY_SIZE / 2;

	while (i--)
		buffer[i] = sbox[buffer[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
void AES::byte_sub_inv(unsigned char* buffer)
{
	register unsigned char i = KEY_SIZE / 2;

	while (i--)
		buffer[i] = sboxinv[buffer[i]];
}

// Shift rows - can be parallel
void AES::shift_rows(unsigned char* buffer)
{
	register unsigned char i, j, k, l; 

	i = buffer[1];
	buffer[1] = buffer[5];
	buffer[5] = buffer[9];
	buffer[9] = buffer[13];
	buffer[13] = i;

	j = buffer[10];
	buffer[10] = buffer[2];
	buffer[2] = j;

	k = buffer[3];
	buffer[3] = buffer[15];
	buffer[15] = buffer[11];
	buffer[11] = buffer[7];
	buffer[7] = k;

	l = buffer[14];
	buffer[14] = buffer[6];
	buffer[6] = l;
}

// Inverse shift rows - can be parallel
void AES::shift_rows_inv(unsigned char* buffer)
{
	register unsigned char i, j, k, l; 

	i = buffer[1];
	buffer[1] = buffer[13];
	buffer[13] = buffer[9];
	buffer[9] = buffer[5];
	buffer[5] = i;

	j = buffer[2];
	buffer[2] = buffer[10];
	buffer[10] = j;

	k = buffer[3];
	buffer[3] = buffer[7];
	buffer[7] = buffer[11];
	buffer[11] = buffer[15];
	buffer[15] = k;

	l = buffer[6];
	buffer[6] = buffer[14];
	buffer[14] = l;
}

// Mix column
void AES::mix_columns(unsigned char* buffer)
{
	register unsigned char i, a, b, c, d, e;

	for (i = 0; i < 16; i += 4)
	{
		a = buffer[i];
		b = buffer[i + 1];
		c = buffer[i + 2];
		d = buffer[i + 3];

		e = a ^ b ^ c ^ d;

		buffer[i]	  ^= e ^ reduction(a^b);
		buffer[i + 1] ^= e ^ reduction(b^c);
		buffer[i + 2] ^= e ^ reduction(c^d);
		buffer[i + 3] ^= e ^ reduction(d^a);
	}
}

// Inverse mix column
void AES::mix_columns_inv(unsigned char* buffer)
{
	register unsigned char i, a, b, c, d, e, x, y, z;

	for (i = 0; i < 16; i += 4)
	{
		a = buffer[i];
		b = buffer[i + 1];
		c = buffer[i + 2];
		d = buffer[i + 3];

		e = a ^ b ^ c ^ d;
		z = reduction(e);
		x = e ^ reduction(reduction(z^a^c));  y = e ^ reduction(reduction(z^b^d));

		buffer[i]	  ^= x ^ reduction(a^b);
		buffer[i + 1] ^= y ^ reduction(b^c);
		buffer[i + 2] ^= x ^ reduction(c^d);
		buffer[i + 3] ^= y ^ reduction(d^a);
	}
}

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Modulo operation on galois field polynom
inline unsigned char reduction(unsigned char x)
{
	return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}