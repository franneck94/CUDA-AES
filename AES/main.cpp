/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>

#include "Helper.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;

int main()
{
	AES aes;
	aes.key_schedule();

	getchar();
}