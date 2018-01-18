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
	#define KEY_BLOCK 24 
#endif  
#if AES_BITS == 256 
	#define NUM_ROUNDS 14
	#define SUB_KEYS (NUM_ROUNDS + 1)
	#define KEY_BLOCK 32
#endif  

using std::vector;


/*********************************************************************/
/*                          DEVICE DECLARATIONS                      */
/*********************************************************************/

__device__ unsigned char *d_keySchedule;

__device__ void byte_sub(unsigned char *internBuffer, unsigned char *sharedSbox);
__device__ void shift_rows(unsigned char *internBuffer);
__device__ void mix_columns(unsigned char* column);
__device__ void key_addition(unsigned char *internBuffer, unsigned char *key, const unsigned int &round);
__device__ unsigned char mulGaloisField2_8(unsigned char a, unsigned char b);

/*********************************************************************/
/*                          KERNEL DECLARATIONS                      */
/*********************************************************************/

__global__ void aes_encryption(unsigned char* SBOX, unsigned char* BufferData, unsigned char* SubKeys);

/*********************************************************************/
/*                        HOST KEY FUNCTIONS                         */
/*********************************************************************/

unsigned char* key_schedule(unsigned char *key);
unsigned char* sub_key128(unsigned char *prev_subkey, const int &r);