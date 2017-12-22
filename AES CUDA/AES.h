#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>

#include "Table.h"

#define AES_BITS 128

#if AES_BITS == 128  
	#define NUM_ROUNDS 10
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 16
#endif  
#if AES_BITS == 192  
	#define NUM_ROUNDS 12
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 16 
#endif  
#if AES_BITS == 256 
	#define NUM_ROUNDS 14
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 16
#endif   

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
	ByteArray encrypt(ByteArray &message);
	ByteArray decrypt(ByteArray &message);

	// Key schedule functions
	void key_schedule();
	ByteArray sub_key128(ByteArray &prev_subkey, const int &r);

	// Sub-Layers of AES round
	// Byte-Sub
	void byte_sub(ByteArray &message);
	void byte_sub_inv(ByteArray &message);
	// SHift-Rows
	void shift_rows(ByteArray &message);
	void shift_rows_inv(ByteArray &message);
	// Mix-Col
	void mix_columns(ByteArray &message);
	void mix_columns_inv(ByteArray &message);
	// Key-Add
	void key_addition(ByteArray &message, const int &r);
};