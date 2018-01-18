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

__device__ void byte_sub(unsigned char *message);
__device__ void byte_sub_inv(unsigned char *message);
__device__ void shift_rows(unsigned char *message);
__device__ void shift_rows_inv(unsigned char *message);
__device__ void mix_columns(unsigned char *message);
__device__ void mix_columns_inv(unsigned char *message);
__device__ void key_addition(unsigned char *message, unsigned char *key, const unsigned int &start);

__device__ void aes_encrypt_ctr(unsigned char *in, unsigned char *out, unsigned char *key, int counter);
__device__ void aes_encrypt(unsigned char *out, unsigned char *key);
__device__ void aes_decrypt(unsigned char *out, unsigned char *key);

/*********************************************************************/
/*                          KERNEL DECLARATIONS                      */
/*********************************************************************/

__global__ void cuda_aes_decrypt(unsigned char *in, unsigned char *out, int n);
__global__ void cuda_aes_encrypt(unsigned char *in, unsigned char *out, int n);
__global__ void cuda_aes_encrypt_ctr(unsigned char *in, unsigned char *out, int n, int counter);

/*********************************************************************/
/*                        HOST KEY FUNCTIONS                         */
/*********************************************************************/

unsigned char* key_schedule(unsigned char *key);
unsigned char* sub_key128(unsigned char *prev_subkey, const int &r);