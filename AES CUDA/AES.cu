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

	m_key = new unsigned char [ 0xde, 0xca, 0xfb, 0xad,
								0xc0, 0xde, 0xba, 0x5e,
								0xde, 0xad, 0xc0, 0xde,
								0xba, 0xdc, 0x0d, 0xed ];

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
unsigned char* AES::encrypt(unsigned char *message)
{
	register int round = 0;

	// Key-Add before round 1 (R0)
	key_addition(message, round);
	round = 1;

	// Round 1 to NUM_ROUNDS - 1 (R1 to R9)
	for (round; round != NUM_ROUNDS; round++)
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
unsigned char* AES::decrypt(unsigned char *message)
{
	register int round = NUM_ROUNDS;

	// Key-Add before round (Inverse NUM_ROUNDS)
	key_addition(message, round);
	shift_rows_inv(message);
	byte_sub_inv(message);
	round = NUM_ROUNDS - 1;

	// Round NUM_ROUNDS - 1 to 1 (Inverse R9 to R1)
	for (round; round > 0; round--)
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
unsigned char* AES::sub_key128(unsigned char *prev_subkey, const int &r)
{
	unsigned char *result;
	result = new unsigned char[KEY_BLOCK];

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
void AES::byte_sub(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_sbox = sizeof(sbox);
	unsigned char *d_sbox;
	cudaMalloc((void **)&d_sbox, size_sbox);
	cudaMemcpy(d_sbox, sbox, size_sbox, cudaMemcpyHostToDevice);

	//<<<1, 1>>> byte_sub_kernel(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_sbox);
}

// Inverse byte substitution (S-Boxes) can be parallel
void AES::byte_sub_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_sboxinv = sizeof(sboxinv);
	unsigned char *d_sboxinv;
	cudaMalloc((void **)&d_sboxinv, size_sboxinv);
	cudaMemcpy(d_sboxinv, sboxinv, size_sboxinv, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> byte_sub_inv_kernel(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_sboxinv);
}

// Shift rows - can be parallel
// B0, B4, B8, B12 stays the same
void AES::shift_rows(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> shift_rows_kernel(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Inverse shift rows - can be parallel
// C0, C4, C8, C12 stays the same
void AES::shift_rows_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> shift_rows_inv_kernel(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Mix column - can be parallel
void AES::mix_columns(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_ltable = sizeof(ltable);
	unsigned char *d_ltable;
	cudaMalloc((void **)&d_ltable, size_ltable);
	cudaMemcpy(d_ltable, ltable, size_ltable, cudaMemcpyHostToDevice);

	int size_atable = sizeof(atable);
	unsigned char *d_atable;
	cudaMalloc((void **)&d_atable, size_atable);
	cudaMemcpy(d_atable, atable, size_atable, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> mix_columns_kernel(d_message, ltable, atable);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

// Inverse mix column
void AES::mix_columns_inv(unsigned char *message)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_ltable = sizeof(ltable);
	unsigned char *d_ltable;
	cudaMalloc((void **)&d_ltable, size_ltable);
	cudaMemcpy(d_ltable, ltable, size_ltable, cudaMemcpyHostToDevice);

	int size_atable = sizeof(atable);
	unsigned char *d_atable;
	cudaMalloc((void **)&d_atable, size_atable);
	cudaMemcpy(d_atable, atable, size_atable, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> mix_columns_inv_kernel(d_message);

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
}

void AES::key_addition(unsigned char *message, const int &r)
{
	int size_message = sizeof(message);
	unsigned char *d_message;
	cudaMalloc((void **)&d_message, size_message);
	cudaMemcpy(d_message, message, size_message, cudaMemcpyHostToDevice);

	int size_subkey = sizeof(m_subkeys[r]);
	unsigned char *d_subkey = &m_subkeys[r][0];
	cudaMalloc((void **)&d_subkey, size_subkey);
	cudaMemcpy(d_subkey, &m_subkeys[r], size_subkey, cudaMemcpyHostToDevice);

	//<<<1, 1 >>> key_addition_kernel(d_message, d_subkey, message->size());

	cudaMemcpy(message, d_message, size_message, cudaMemcpyDeviceToHost);
	cudaFree(d_message);
	cudaFree(d_subkey);
}