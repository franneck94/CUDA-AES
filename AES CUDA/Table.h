#pragma once

/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

/*********************************************************************/
/*                            HOST TABLES                            */
/*********************************************************************/

extern const unsigned char RC[10];
extern unsigned char h_sbox[256];