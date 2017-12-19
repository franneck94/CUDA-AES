/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "AES.hpp"
#include "Helper.hpp"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                           CONSTRUCTORS                            */
/*********************************************************************/

// Constructor of AES en/decryption
AES::AES() : m_subkeys(SUB_KEYS)
{
	cout << "AES datastream created!" << endl;

	m_message = { 0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
				0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d,
				0x50, 0x86, 0xcb, 0x9b, 0x50, 0x72, 0x19, 0xee,
				0x95, 0xdb, 0x11, 0x3a, 0x91, 0x76, 0x78, 0xb2,
				0x73, 0xbe, 0xd6, 0xb8, 0xe3, 0xc1, 0x74, 0x3b,
				0x71, 0x16, 0xe6, 0x9e, 0x22, 0x22, 0x95, 0x16,
				0x3f, 0xf1, 0xca, 0xa1, 0x68, 0x1f, 0xac, 0x09,
				0x12, 0x0e, 0xca, 0x30, 0x75, 0x86, 0xe1, 0xa7 };

	m_key = { 0xde, 0xca, 0xfb, 0xad,
			0xc0, 0xde, 0xba, 0x5e,
			0xde, 0xad, 0xc0, 0xde,
			0xba, 0xdc, 0x0d, 0xed};
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

/*********************************************************************/
/*                           KEY FUNCTIONS                           */
/*********************************************************************/

// Computing the round keys
void AES::key_schedule()
{
	cout << "Starting key schedule!" << endl;

	for (int r = 0; r < SUB_KEYS; r++)
	{
		cout << "Key schedule round: " << std::dec << r << endl;
		cout << "With subkey: " << endl;

		if (r == 0)
			m_subkeys[r] = m_key;
		else
			m_subkeys[r] = sub_key(m_subkeys[r - 1],  r - 1);
		
		print_byte_array(m_subkeys[r]);
	}
}

// Computing subkeys for round 1 up to 10
ByteArray AES::sub_key(ByteArray &prev_subkey, const int  &r)
{
	ByteArray result(KEY_SIZE);
	register int i;

	result[0] = xor (prev_subkey[0], xor (sbox[prev_subkey[13]], RC[r]));
	result[1] = xor (prev_subkey[1], sbox[prev_subkey[14]]);
	result[2] = xor (prev_subkey[2], sbox[prev_subkey[15]]);
	result[3] = xor (prev_subkey[3], sbox[prev_subkey[12]]);

	for (i = 4; i != result.size(); i += 4)
	{
		result[i + 0] = xor (result[i + 0 - 4], prev_subkey[i + 0]);
		result[i + 1] = xor (result[i + 1 - 4], prev_subkey[i + 1]);
		result[i + 2] = xor (result[i + 2 - 4], prev_subkey[i + 2]);
		result[i + 3] = xor (result[i + 3 - 4], prev_subkey[i + 3]);
	}

	return result;
}

/*********************************************************************/
/*                              SUB LAYER                            */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
void AES::byte_sub(ByteArray &buffer)
{
	register int i = 0;

	for (i; i < KEY_SIZE; ++i)
		buffer[i] = sbox[buffer[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
void AES::byte_sub_inv(ByteArray &buffer)
{
	register int i = 0;

	for (i; i < KEY_SIZE; ++i)
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

void AES::key_addition(unsigned char *buffer)
{

}