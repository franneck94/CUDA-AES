#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>

#include "Table.h"

#define AES_BITS 128
#define BLOCKS_PER_LAUNCH 5
#define THREADS_PER_BLOCK 512

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
/*                          DEVICE DECLARATIONS                      */
/*********************************************************************/

__device__ unsigned char *d_keySchedule;

__device__ void byte_sub_kernel(unsigned char *message);
__device__ void byte_sub_inv_kernel(unsigned char *message);
__device__ void shift_rows_kernel(unsigned char *message);
__device__ void shift_rows_inv_kernel(unsigned char *message);
__device__ void mix_columns_kernel(unsigned char *message);
__device__ void mix_columns_inv_kernel(unsigned char *message);
__device__ void key_addition_kernel(unsigned char *message, unsigned char **subkeys, const unsigned int &round);

/*********************************************************************/
/*                          KERNEL DECLARATIONS                      */
/*********************************************************************/

__device__ void aes_encrypt_ctr(const unsigned char in[], unsigned char out[],
							const unsigned char key[], int keysize, int counter);
__device__ void aes_encrypt(const unsigned char in[], unsigned char out[],
							const unsigned char key[], int keysize);
__device__ void aes_decrypt(const unsigned char in[], unsigned char out[],
							const unsigned char key[], int keysize);

/*********************************************************************/
/*                          HOST DECLARATIONS                       */
/*********************************************************************/

void launchKernel(char *inFileName, char *outFileName);
void aes_key_setup(const unsigned char key[], unsigned char w[], int keysize);
int getKeySchedule(char *keyFilename);