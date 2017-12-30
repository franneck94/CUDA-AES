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
	#define SHIFT_ROW_LIMIT 3
	#define MIX_COLUMN_LIMIT 4
#endif  
#if AES_BITS == 192  
	#define NUM_ROUNDS 12
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 24 
	#define SHIFT_ROW_LIMIT 3
	#define MIX_COLUMN_LIMIT 6
#endif  
#if AES_BITS == 256 
	#define NUM_ROUNDS 14
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 32
	#define SHIFT_ROW_LIMIT 3
	#define MIX_COLUMN_LIMIT 8
#endif  

using std::vector;

/*********************************************************************/
/*                         CLASS DEFINITION                          */
/*********************************************************************/

class AES
{

public:
	// Constructor
	AES(unsigned char *key);

	// Main functions of AES
	unsigned char* encrypt(unsigned char *message);
	unsigned char* decrypt(unsigned char *message);

private:
	// Member vairables
	unsigned char *m_key;
	vector<unsigned char*> m_subkeys;

	// Key schedule functions
	void key_schedule();
	unsigned char* sub_key128(unsigned char *prev_subkey, const int &r);

	// Sub-Layers of AES round
	// Byte-Sub
	void byte_sub(unsigned char *message);
	void byte_sub_inv(unsigned char *message);
	// Shift-Rows
	void shift_rows(unsigned char *message);
	void shift_rows_inv(unsigned char *message);
	// Mix-Col
	void mix_columns(unsigned char *message);
	void mix_columns_inv(unsigned char *message);
	// Key-Add
	void key_addition(unsigned char *message, const int &r);
};