/**Translate OpenCL error codes to strings.
 * 
 * Source: http://web.engr.oregonstate.edu/~mjb/cs575e/
 */
#include <gpucf/cl.h>

#include <string>

std::string PrintCLError(cl_int);
