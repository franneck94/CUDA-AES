#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <omp.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "config.h"

#include "timer.h"

using namespace std;

static unsigned int nvtimer;

BYTE SubKeys[176];
/* Test Key */
BYTE Key[16] = {
	0x0f, 0x15, 0x71, 0xc9, 0x47, 0xd9, 0xe8, 0x59, 0x0c, 0xb7, 0xad, 0xd6, 0xaf, 0x7f, 0x67, 0x98
};


/* START AES ENCRYPTION FUNCTIONS */
const BYTE Sbox[256] = 
{
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__device__ void aes_KeyAddition(BYTE* internBuffer, BYTE* SubKeys, int round) {
	for (register int k = 0; k<16; k++) {
		internBuffer[k] = internBuffer[k] ^ SubKeys[(16 * round) + k];
	}
}

__device__ void aes_SubstitutionBox(BYTE* internBuffer, BYTE* Sbox) {
	for (register int k = 0; k<16; k++) {
		internBuffer[k] = Sbox[internBuffer[k]];
	}
}

__device__ void aes_ShiftRows(BYTE* internBuffer) {
	register BYTE tmpBuffer;
	
	tmpBuffer = internBuffer[1];
	internBuffer[1] = internBuffer[5];
	internBuffer[5] = internBuffer[9];
	internBuffer[9] = internBuffer[13];
	internBuffer[13] = tmpBuffer;

	tmpBuffer = internBuffer[2];
	internBuffer[2] = internBuffer[10];
	internBuffer[10] = tmpBuffer;
	tmpBuffer = internBuffer[6];
	internBuffer[6] = internBuffer[14];
	internBuffer[14] = tmpBuffer;

	tmpBuffer = internBuffer[15];
	internBuffer[15] = internBuffer[11];
	internBuffer[11] = internBuffer[7];
	internBuffer[7] = internBuffer[3];
	internBuffer[3] = tmpBuffer;
}

__device__ BYTE mulGaloisField2_8(BYTE a, BYTE b) 
{
	register BYTE p = 0;
	register BYTE hi_bit_set;
	register BYTE counter;

	for (counter = 0; counter < 8; counter++) 
	{
		if ((b & 1) == 1)
			p ^= a;
		hi_bit_set = (a & 0x80);
		a <<= 1;
		if (hi_bit_set == 0x80)
			a ^= 0x1b;
		b >>= 1;
	}

	return p;
}

__device__ void mixColumn(BYTE* column) 
{
	register BYTE i;
	register BYTE cpy[4];

	for (i = 0; i < 4; i++)
	{
		cpy[i] = column[i];
	}

	column[0] = mulGaloisField2_8(cpy[0], 2) ^
		mulGaloisField2_8(cpy[1], 3) ^
		mulGaloisField2_8(cpy[2], 1) ^
		mulGaloisField2_8(cpy[3], 1);

	column[1] = mulGaloisField2_8(cpy[0], 1) ^
		mulGaloisField2_8(cpy[1], 2) ^
		mulGaloisField2_8(cpy[2], 3) ^
		mulGaloisField2_8(cpy[3], 1);

	column[2] = mulGaloisField2_8(cpy[0], 1) ^
		mulGaloisField2_8(cpy[1], 1) ^
		mulGaloisField2_8(cpy[2], 2) ^
		mulGaloisField2_8(cpy[3], 3);

	column[3] = mulGaloisField2_8(cpy[0], 3) ^
		mulGaloisField2_8(cpy[1], 1) ^
		mulGaloisField2_8(cpy[2], 1) ^
		mulGaloisField2_8(cpy[3], 2);
}

__device__ void aes_MixColumns(BYTE* internBuffer) 
{
	register int i, j;
	register BYTE column[4];

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++) 
		{
			column[j] = internBuffer[(i * 4) + j];
		}

		mixColumn(column);

		for (j = 0; j < 4; j++) 
		{
			internBuffer[(i * 4) + j] = column[j];
		}
	}
}

__global__ void aes_encryption(BYTE* SBOX, BYTE* BufferData, BYTE* SubKeys)
{
	/* each thread progresses 16 bytes */
	register int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;

	// IF threadId == 0, copy shared S_BOX & copy shared SubKeys
	__shared__ BYTE sharedSbox[256];
	__shared__ BYTE sharedSubKeys[176];
	if (threadIdx.x == 0) 
	{
		for (int i = 0; i<256; i++) 
		{
			sharedSbox[i] = SBOX[i];
		}
		for (int i = 0; i<176; i++)
		{
			sharedSubKeys[i] = SubKeys[i];
		}
	}
	__syncthreads();

	/* Copy 16 bytes to intern buffer */
	register BYTE internBuffer[16];
	for (register int i = 0; i<16; i++) 
	{
		internBuffer[i] = BufferData[id + i];
	}

	/*Initial XOR */
	aes_KeyAddition(internBuffer, sharedSubKeys, 0);


	/* Round loop */
	for (register int i = 1; i<11; i++) 
	{
		aes_SubstitutionBox(internBuffer, sharedSbox);
		aes_ShiftRows(internBuffer);

		if (i != 10) 
		{
			aes_MixColumns(internBuffer);
		}

		aes_KeyAddition(internBuffer, sharedSubKeys, i);
	}

	/* Copy everything back to the buffer */
	for (register int i = 0; i<16; i++)
	{
		BufferData[id + i] = internBuffer[i];
	}
}
/* END AES ENCRYPTION FUNCTIONS */

/* START AES DECRYPTION FUNCTIONS */
const BYTE InvSbox[256] = {
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

__device__ void aes_InvSubstitutionBox(BYTE* internBuffer, BYTE* InvSbox) {
	for (register int k = 0; k<16; k++) {
		internBuffer[k] = InvSbox[internBuffer[k]];
	}
}

__device__ void aes_InvShiftRows(BYTE* internBuffer) {
	register BYTE tmpBuffer;

	tmpBuffer = internBuffer[13];
	internBuffer[13] = internBuffer[9];
	internBuffer[9] = internBuffer[5];
	internBuffer[5] = internBuffer[1];
	internBuffer[1] = tmpBuffer;

	tmpBuffer = internBuffer[2];
	internBuffer[2] = internBuffer[10];
	internBuffer[10] = tmpBuffer;
	tmpBuffer = internBuffer[6];
	internBuffer[6] = internBuffer[14];
	internBuffer[14] = tmpBuffer;

	tmpBuffer = internBuffer[3];
	internBuffer[3] = internBuffer[7];
	internBuffer[7] = internBuffer[11];
	internBuffer[11] = internBuffer[15];
	internBuffer[15] = tmpBuffer;

}


__device__ void InvMixColumn(BYTE* column) {
	register BYTE i;
	register BYTE cpy[4];

	for (i = 0; i < 4; i++) {
		cpy[i] = column[i];
	}

	column[0] = mulGaloisField2_8(cpy[0], 0x0E) ^
		mulGaloisField2_8(cpy[1], 0x0B) ^
		mulGaloisField2_8(cpy[2], 0x0D) ^
		mulGaloisField2_8(cpy[3], 0x09);

	column[1] = mulGaloisField2_8(cpy[0], 0x09) ^
		mulGaloisField2_8(cpy[1], 0x0E) ^
		mulGaloisField2_8(cpy[2], 0x0B) ^
		mulGaloisField2_8(cpy[3], 0x0D);

	column[2] = mulGaloisField2_8(cpy[0], 0x0D) ^
		mulGaloisField2_8(cpy[1], 0x09) ^
		mulGaloisField2_8(cpy[2], 0x0E) ^
		mulGaloisField2_8(cpy[3], 0x0B);

	column[3] = mulGaloisField2_8(cpy[0], 0x0B) ^
		mulGaloisField2_8(cpy[1], 0x0D) ^
		mulGaloisField2_8(cpy[2], 0x09) ^
		mulGaloisField2_8(cpy[3], 0x0E);
}

__device__ void aes_InvMixColumns(BYTE* internBuffer) {
	register int i, j;
	register BYTE column[4];

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			column[j] = internBuffer[(i * 4) + j];
		}

		InvMixColumn(column);

		for (j = 0; j < 4; j++) {
			internBuffer[(i * 4) + j] = column[j];
		}
	}
}

__global__ void aes_decryption(BYTE* InvSbox, BYTE* BufferData, BYTE* SubKeys) {
	/* each thread progresses 16 bytes */
	register int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;

	/* IF threadId == 0, copy shared S_BOX & copy shared SubKeys
	* Wait for Sync.
	*/
	__shared__ BYTE sharedInvSbox[256];
	__shared__ BYTE sharedSubKeys[176];
	if (threadIdx.x == 0) {
		for (int i = 0; i<256; i++) {
			sharedInvSbox[i] = InvSbox[i];
		}
		for (int i = 0; i<176; i++) {
			sharedSubKeys[i] = SubKeys[i];
		}
	}
	__syncthreads();

	/* Now we are synced with all threads */
	/* Copy 16 bytes to intern buffer */
	register BYTE internBuffer[16];
	for (register int i = 0; i<16; i++) {
		internBuffer[i] = BufferData[id + i];
	}
	/* Now run complete AES */

	/*Initial XOR */
	aes_KeyAddition(internBuffer, sharedSubKeys, 10);

	/* Round loop */
	for (register int i = 9; i >= 0; i--) {

		/* Inverted ShiftRows */
		aes_InvShiftRows(internBuffer);

		/* Inverted SubBytes */
		aes_InvSubstitutionBox(internBuffer, sharedInvSbox);

		/* Key Addition */
		aes_KeyAddition(internBuffer, sharedSubKeys, i);

		if (i != 0) {
			/* Inverted MixColumns, skip in last round */
			aes_InvMixColumns(internBuffer);
		}
	}

	/* Copy everything back to the buffer */
	for (register int i = 0; i<16; i++) {
		BufferData[id + i] = internBuffer[i];
	}
}
/* END AES DECRYPTION FUNCTIONS */

/* START AES KEYSCHEDULE */
BYTE Rcon[11] = {
	0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

void aes_keyschedule_gFunction(BYTE * input, int round) {
	/* G Function in AES Keyschedule */

	/* ROTATE / 1 SHIFT LEFT */
	BYTE tmpBuffer;
	tmpBuffer = input[0];
	input[0] = input[1];
	input[1] = input[2];
	input[2] = input[3];
	input[3] = tmpBuffer;

	/* SBOX */
	input[0] = Sbox[input[0]];
	input[1] = Sbox[input[1]];
	input[2] = Sbox[input[2]];
	input[3] = Sbox[input[3]];

	/* XOR Rcon */
	input[0] ^= Rcon[round];
}

void aes_keyschedule(BYTE * Key, BYTE * SubKeys) {
	/* First SubKey = Key */
	BYTE Buffer[16];
	for (int i = 0; i<16; i++) {
		SubKeys[i] = Key[i];
		Buffer[i] = Key[i];
	}
	/* Calculate remaining SubKeys */
	for (int round = 1; round<11; round++) {
		/* Buffer for g function */
		BYTE g_Buffer[4];
		for (int i = 0; i<4; i++) {
			g_Buffer[i] = Buffer[12 + i];
		}
		aes_keyschedule_gFunction(g_Buffer, round);
		/* BUFFER[0] XOR Output_G (32 bit) */
		for (int i = 0; i<4; i++) {
			Buffer[i] ^= g_Buffer[i];
		}

		/* Buffer[0] XOR BUFFER[1] (32-bit) */
		for (int i = 0; i<4; i++) {
			Buffer[4 + i] ^= Buffer[i];
		}

		/* Buffer[1] XOR BUFFER[2] (32-bit) */
		for (int i = 0; i<4; i++) {
			Buffer[8 + i] ^= Buffer[4 + i];
		}

		/* Buffer[2] XOR BUFFER[3] (32-bit) */
		for (int i = 0; i<4; i++) {
			Buffer[12 + i] ^= Buffer[8 + i];
		}

		for (int i = 0; i<16; i++) {
			SubKeys[(round * 16) + i] = Buffer[i];
		}
	}

}

/* END AES KEYSCHEDULE */


static void printByteArray(BYTE byteArray[], int length) {
	printf("Output:\n");
	for (int i = 0; i<length; i++) {
		printf("%02X ", byteArray[i]);
	}
	printf("\n");
}

static void printAsState(BYTE byteArray[], int length) {
	printf("Output:\n");
	for (int i = 0; i<4; i++) {
		printf("%02X ", byteArray[i]);
		printf("%02X ", byteArray[i + 4]);
		printf("%02X ", byteArray[i + 8]);
		printf("%02X\n", byteArray[i + 12]);
	}
	printf("\n");
}

bool file_exists(const char * filename)
{
	if (FILE * file = fopen(filename, "r"))
	{
		fclose(file);
		return true;
	}
	return false;
}

int main(int argc, char** argv) {

	/* Decryption or Encryption */
	int mode;
	mode = 0;


	printf("Reading in file..\n");
	/* Open input file */
	FILE * fp_input;
	fp_input = fopen("C:/Users/Jan/Dropbox/Master AI/Parallel Computing/Project/text2.txt", "rb");


	/* Check filesize */
	fseek(fp_input, 0L, SEEK_END);
	long int FILESIZE = ftell(fp_input);

	fseek(fp_input, 0L, SEEK_SET);

	/* Read in input file */
	BYTE * plaintexts = (BYTE *)malloc(sizeof(BYTE)*FILESIZE);

	/* Call Function */
	int ThreadsPerBlock = 1024;
	int Blocks = (FILESIZE / 16) / ThreadsPerBlock;

	unsigned long input_cnt = 0;
	while (feof(fp_input) == 0) {
		plaintexts[input_cnt] = fgetc(fp_input);
		input_cnt++;
	}
	fclose(fp_input);

	printf("Read in %lu Bytes\n", input_cnt);
	/* Read in done */

	/* AES KEY SCHEDULE */
	aes_keyschedule(Key, SubKeys);

	for (int i = 0; i<11; i++) {
		printf("\nRoundKey %i\t", i);
		for (int j = 0; j<16; j++) {
			printf("%02X ", SubKeys[(16 * i) + j]);
		}
	}
	printf("\nCalculating AES..\n");

	/* Copy everything to the card */
	/* Copy Key */
	//Copy encryption key to device
	BYTE* buffer_key;
	cudaMalloc((void **)&buffer_key, sizeof(BYTE) * 176);
	cudaMemcpy(buffer_key, SubKeys, sizeof(BYTE) * 176, cudaMemcpyHostToDevice);

	/* Copy SBOX */
	BYTE* buffer_sbox;
	cudaMalloc((void **)&buffer_sbox, sizeof(BYTE) * 256);
	if (mode == 0) {
		cudaMemcpy(buffer_sbox, Sbox, sizeof(BYTE) * 256, cudaMemcpyHostToDevice);
	}
	else {
		cudaMemcpy(buffer_sbox, InvSbox, sizeof(BYTE) * 256, cudaMemcpyHostToDevice);
	}

	/* Copy plaintexts */
	BYTE* buffer_Data;
	cudaMalloc((void **)&buffer_Data, sizeof(BYTE)*FILESIZE);
	cudaMemcpy(buffer_Data, plaintexts, sizeof(BYTE)*FILESIZE, cudaMemcpyHostToDevice);
	/*Copy done */

	GpuTimer timer;
	timer.Start();

	if (mode == 0) 
	{
		aes_encryption << < Blocks, ThreadsPerBlock >> > (buffer_sbox, buffer_Data, buffer_key);
	}
	else 
	{
		aes_decryption << < Blocks, ThreadsPerBlock >> > (buffer_sbox, buffer_Data, buffer_key);
	}

	timer.Stop();
	float milliseconds = timer.ElapsedMilliSeconds();
	printf("MS: %f\n", milliseconds);

	cudaError_t code = cudaPeekAtLastError();

	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
		getchar();
		if (abort) exit(code);
	}


	/* And Synchronize */
	cudaThreadSynchronize();

	BYTE *returnBuffer = (BYTE *)malloc(sizeof(BYTE)*FILESIZE);
	cudaMemcpy(returnBuffer, buffer_Data, sizeof(BYTE)*FILESIZE, cudaMemcpyDeviceToHost);

	cudaFree(buffer_sbox);
	cudaFree(buffer_Data);
	cudaFree(buffer_key);
	printAsState(returnBuffer, 16);

	getchar();
}
