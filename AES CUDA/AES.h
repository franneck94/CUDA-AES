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
/*                            DECLARATIONS                           */
/*********************************************************************/

void encrypt(unsigned char *message, unsigned char **subkeys);
void decrypt(unsigned char *message, unsigned char **subkeys);

__global__ void byte_sub_kernel(unsigned char *message);
__global__ void byte_sub_inv_kernel(unsigned char *message);
__global__ void shift_rows_kernel(unsigned char *message);
__global__ void shift_rows_inv_kernel(unsigned char *message);
__global__ void mix_columns_kernel(unsigned char *message);
__global__ void mix_columns_inv_kernel(unsigned char *message);
__global__ void key_addition_kernel(unsigned char *message, unsigned char **subkeys, const unsigned int &round);



/*********************************************************************/
/*                         CLASS DEFINITION                          */
/*********************************************************************/

class AES
{

public:
	// Constructor
	AES(unsigned char *key);
	unsigned char** get_subkeys();

private:
	// Member vairables
	unsigned char *m_key;
	unsigned char** m_subkeys;

	// Key schedule functions
	void key_schedule();
	unsigned char* sub_key128(unsigned char *prev_subkey, const int &r);
};