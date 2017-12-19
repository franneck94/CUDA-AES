/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#pragma once
#include <vector>
#include <iostream>

#include "Table.hpp"

#define SUB_KEYS 11
#define NUM_ROUNDS 10
#define KEY_SIZE 16

typedef std::vector<unsigned char> ByteArray;
using std::vector;

/*********************************************************************/
/*                         CLASS DEFINITION                          */
/*********************************************************************/

class AES
{

public:
	AES();
	~AES();

//private:
	// Member vairables
	ByteArray m_key;
	vector<ByteArray> m_subkeys;
	ByteArray m_message;
	ByteArray m_encrypted_message;
	ByteArray m_decrypted_message;

	// Main functions of AES
	void encrypt(unsigned char *buffer);
	void decrypt(unsigned char *buffer);

	// Key schedule functions
	void key_schedule();
	ByteArray sub_key(ByteArray &prev_subkey, const int  &r);

	// Sub-Layers of AES round
	// Byte-Sub
	void byte_sub(ByteArray &buffer);
	void byte_sub_inv(ByteArray &buffer);
	// SHift-Rows
	void shift_rows(unsigned char *buffer);
	void shift_rows_inv(unsigned char *buffer);
	// Mix-Col
	void mix_columns(unsigned char *buffer);
	void mix_columns_inv(unsigned char *buffer);
	// Key-Add
	void key_addition(unsigned char *buffer);
};