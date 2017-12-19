/*********************************************************************/
/*                       INCLUDES AND DEFINES                        */
/*********************************************************************/

#include <vector>
#include <iostream>

#include "Helper.hpp"

using std::cout;
using std::endl;
using std::vector;

/*********************************************************************/
/*                          HELPER FUNCTIONS                         */
/*********************************************************************/

// Cout whole ByteArray
void print_byte_array(ByteArray &arr)
{
	for (size_t i = 0; i != arr.size(); ++i)
	{
		cout << std::hex << (int)arr[i] << "\t";
	}
	cout << endl << endl;
}


// Cout hex byte
void print_byte(const unsigned char &byte)
{
	cout << endl << "Byte: " << std::hex << (int)byte;
}