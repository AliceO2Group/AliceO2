/**Translate OpenCL error codes to strings.
 * 
 * Source: http://web.engr.oregonstate.edu/~mjb/cs575e/
 */
#include <CL/cl.h>

const char *PrintCLError(cl_int);
