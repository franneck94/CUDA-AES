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
		cout << endl << std::hex << (int)arr[i];
		if (i % 2 == 0)
			cout << "\t";
	}
	cout << endl;
}