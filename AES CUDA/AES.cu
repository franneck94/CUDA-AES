/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include "AES.h"
#include "Helper.h"

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
				0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d};

	m_key = { 0xde, 0xca, 0xfb, 0xad,
			0xc0, 0xde, 0xba, 0x5e,
			0xde, 0xad, 0xc0, 0xde,
			0xba, 0xdc, 0x0d, 0xed};

	key_schedule();
}

// Destructor of AES en/decryption
AES::~AES()
{
	cout << "AES datastream deleted!" << endl;
}


/*********************************************************************/
/*                       EN- DECRYPTION FUNCTIONS                    */
/*********************************************************************/

// Starting the encryption phase
ByteArray AES::encrypt(ByteArray &message)
{
	register int i = 0, round = 0;

	// Key-Add before round 1 (R0)
	key_addition(message, round);
	round = 1;

	// Round 1 to NUM_ROUNDS - 1 (R1 to R9)
	for (round; round != NUM_ROUNDS; ++round)
	{
		byte_sub(message);
		shift_rows(message);
		mix_columns(message);
		key_addition(message, round);
	}

	// Last round without Mix-Column (RNUM_ROUNDS)
	round = NUM_ROUNDS;
	byte_sub(message);
	shift_rows(message);
	key_addition(message, round);

	// Save encrypted message
	m_encrypted_message = message;
	return message;
}

// Starting the decryption phase
ByteArray AES::decrypt(ByteArray &message)
{
	register int i = 0, round = NUM_ROUNDS;

	// Key-Add before round (Inverse NUM_ROUNDS)
	key_addition(message, round);
	shift_rows_inv(message);
	byte_sub_inv(message);
	round = NUM_ROUNDS - 1;

	// Round NUM_ROUNDS - 1 to 1 (Inverse R9 to R1)
	for (round; round > 0; --round)
	{
		key_addition(message, round);
		mix_columns_inv(message);
		shift_rows_inv(message);
		byte_sub_inv(message);
	}

	// Last round without Mix-Column (Inverse R0)
	round = 0;
	key_addition(message, round);

	// Save decrypted message
	m_decrypted_message = message;
	return message;
}

/*********************************************************************/
/*                           KEY FUNCTIONS                           */
/*********************************************************************/

// Computing the round keys
void AES::key_schedule()
{
	register int r;

	for (r = 0; r != SUB_KEYS; r++)
	{
		if (r == 0)
			m_subkeys[r] = m_key;
		else
		{
			if (AES_BITS == 128)
				m_subkeys[r] = sub_key128(m_subkeys[r - 1], r - 1);
			else
				cout << "TODO! 192-bit and 256-bit not implemented yet." << endl;
		}
	}
}

// Computing subkeys for round 1 up to 10
ByteArray AES::sub_key128(ByteArray &prev_subkey, const int &r)
{
	ByteArray result(KEY_BLOCK);
	register int i;

	result[0] = (prev_subkey[0] ^ (sbox[prev_subkey[13]] ^ RC[r]));
	result[1] = (prev_subkey[1] ^ sbox[prev_subkey[14]]);
	result[2] = (prev_subkey[2] ^ sbox[prev_subkey[15]]);
	result[3] = (prev_subkey[3] ^ sbox[prev_subkey[12]]);

	for (i = 4; i != KEY_BLOCK; i += 4)
	{
		result[i + 0] = (result[i - 4] ^ prev_subkey[i + 0]);
		result[i + 1] = (result[i - 3] ^ prev_subkey[i + 1]);
		result[i + 2] = (result[i - 2] ^ prev_subkey[i + 2]);
		result[i + 3] = (result[i - 1] ^ prev_subkey[i + 3]);
	}

	return result;
}

/*********************************************************************/
/*                              SUB LAYER                            */
/*********************************************************************/

// Byte substitution (S-Boxes) can be parallel
void AES::byte_sub(ByteArray &message)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; ++i)
		message[i] = sbox[message[i]];
}

// Inverse byte substitution (S-Boxes) can be parallel
void AES::byte_sub_inv(ByteArray &message)
{
	register int i = 0;

	for (i; i != KEY_BLOCK; ++i)
		message[i] = sboxinv[message[i]];
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
void AES::shift_rows(ByteArray &message)
{
	register unsigned char i, j, k, l; 

	i = message[1];
	message[1] = message[5];
	message[5] = message[9];
	message[9] = message[13];
	message[13] = i;

	j = message[10];
	message[10] = message[2];
	message[2] = j;
	l = message[14];
	message[14] = message[6];
	message[6] = l;

	k = message[3];
	message[3] = message[15];
	message[15] = message[11];
	message[11] = message[7];
	message[7] = k;
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
void AES::shift_rows_inv(ByteArray &message)
{
	register unsigned char i, j, k, l; 

	i = message[1];
	message[1] = message[13];
	message[13] = message[9];
	message[9] = message[5];
	message[5] = i;

	j = message[2];
	message[2] = message[10];
	message[10] = j;
	l = message[6];
	message[6] = message[14];
	message[14] = l;

	k = message[3];
	message[3] = message[7];
	message[7] = message[11];
	message[11] = message[15];
	message[15] = k;
}

// Mix column - can be parallel
void AES::mix_columns(ByteArray &message)
{
	register unsigned char b0, b1, b2, b3;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		b0 = message[i + 0];
		b1 = message[i + 1];
		b2 = message[i + 2];
		b3 = message[i + 3];

		// Mix-Col Matrix * b vector
		message[i + 0] = mul(b0, 0x02) ^ mul(b1, 0x03) ^ b2 ^ b3;
		message[i + 1] = b0 ^ mul(b1, 0x02) ^ mul(b2, 0x03) ^ b3;
		message[i + 2] = b0 ^ b1 ^ mul(b2, 0x02) ^ mul(b3, 0x03);
		message[i + 3] = mul(b0, 0x03) ^ b1 ^ b2 ^ mul(b3, 0x02);
	}
}

// Inverse mix column
void AES::mix_columns_inv(ByteArray &message)
{
	register unsigned char c0, c1, c2, c3;
	register int i;

	for (i = 0; i != KEY_BLOCK; i += 4)
	{
		c0 = message[i + 0];
		c1 = message[i + 1];
		c2 = message[i + 2];
		c3 = message[i + 3];

		// Mix-Col Inverse Matrix * c vector
		message[i + 0] = mul(c0, 0x0e) ^ mul(c1, 0x0b) ^ mul(c2, 0x0d) ^ mul(c3, 0x09);
		message[i + 1] = mul(c0, 0x09) ^ mul(c1, 0x0e) ^ mul(c2, 0x0b) ^ mul(c3, 0x0d);
		message[i + 2] = mul(c0, 0x0d) ^ mul(c1, 0x09) ^ mul(c2, 0x0e) ^ mul(c3, 0x0b);
		message[i + 3] = mul(c0, 0x0b) ^ mul(c1, 0x0d) ^ mul(c2, 0x09) ^ mul(c3, 0x0e);
	}
}

void AES::key_addition(ByteArray &message, const int &r)
{
	register int i = 0;

	for (i; i != message.size(); ++i)
		message[i] ^= m_subkeys[r][i];
}