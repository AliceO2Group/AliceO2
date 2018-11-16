/**Translate OpenCL error codes to strings.
 * 
 * Source: http://web.engr.oregonstate.edu/~mjb/cs575e/
 */
#include "errorcodes.h"

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>


struct errorcode
{
	cl_int		statusCode;
	char *		meaning;
}
ErrorCodes[ ] =
{
	{ CL_SUCCESS,				            "Success"   		                    },
	{ CL_DEVICE_NOT_FOUND,			        "Device Not Found"			            },
	{ CL_DEVICE_NOT_AVAILABLE,		        "Device Not Available"			        },
	{ CL_COMPILER_NOT_AVAILABLE,		    "Compiler Not Available"		        },
	{ CL_MEM_OBJECT_ALLOCATION_FAILURE,	    "Memory Object Allocation Failure"	    },
	{ CL_OUT_OF_RESOURCES,			        "Out of resources"			            },
	{ CL_OUT_OF_HOST_MEMORY,		        "Out of Host Memory"			        },
	{ CL_PROFILING_INFO_NOT_AVAILABLE,	    "Profiling Information Not Available"	},
	{ CL_MEM_COPY_OVERLAP,			        "Memory Copy Overlap"			        },
	{ CL_IMAGE_FORMAT_MISMATCH,		        "Image Format Mismatch"			        },
	{ CL_IMAGE_FORMAT_NOT_SUPPORTED,	    "Image Format Not Supported"		    },
	{ CL_BUILD_PROGRAM_FAILURE,		        "Build Program Failure"			        },
	{ CL_MAP_FAILURE,			            "Map Failure"				            },
	{ CL_INVALID_VALUE,			            "Invalid Value"				            },
	{ CL_INVALID_DEVICE_TYPE,		        "Invalid Device Type"			        },
	{ CL_INVALID_PLATFORM,			        "Invalid Platform"			            },
	{ CL_INVALID_DEVICE,			        "Invalid Device"			            },
	{ CL_INVALID_CONTEXT,			        "Invalid Context"			            },
	{ CL_INVALID_QUEUE_PROPERTIES,		    "Invalid Queue Properties"		        },
	{ CL_INVALID_COMMAND_QUEUE,		        "Invalid Command Queue"			        },
	{ CL_INVALID_HOST_PTR,			        "Invalid Host Pointer"			        },
	{ CL_INVALID_MEM_OBJECT,		        "Invalid Memory Object"			        },
	{ CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,	"Invalid Image Format Descriptor"	    },
	{ CL_INVALID_IMAGE_SIZE,		        "Invalid Image Size"			        },
	{ CL_INVALID_SAMPLER,			        "Invalid Sampler"			            },
	{ CL_INVALID_BINARY,			        "Invalid Binary"			            },
	{ CL_INVALID_BUILD_OPTIONS,		        "Invalid Build Options"			        },
	{ CL_INVALID_PROGRAM,			        "Invalid Program"			            },
	{ CL_INVALID_PROGRAM_EXECUTABLE,	    "Invalid Program Executable"		    },
	{ CL_INVALID_KERNEL_NAME,		        "Invalid Kernel Name"			        },
	{ CL_INVALID_KERNEL_DEFINITION,		    "Invalid Kernel Definition"		        },
	{ CL_INVALID_KERNEL,			        "Invalid Kernel"			            },
	{ CL_INVALID_ARG_INDEX,			        "Invalid Argument Index"		        },
	{ CL_INVALID_ARG_VALUE,			        "Invalid Argument Value"		        },
	{ CL_INVALID_ARG_SIZE,			        "Invalid Argument Size"			        },
	{ CL_INVALID_KERNEL_ARGS,		        "Invalid Kernel Arguments"		        },
	{ CL_INVALID_WORK_DIMENSION,		    "Invalid Work Dimension"		        },
	{ CL_INVALID_WORK_GROUP_SIZE,		    "Invalid Work Group Size"		        },
	{ CL_INVALID_WORK_ITEM_SIZE,		    "Invalid Work Item Size"		        },
	{ CL_INVALID_GLOBAL_OFFSET,		        "Invalid Global Offset"			        },
	{ CL_INVALID_EVENT_WAIT_LIST,		    "Invalid Event Wait List"		        },
	{ CL_INVALID_EVENT,			            "Invalid Event"				            },
	{ CL_INVALID_OPERATION,			        "Invalid Operation"			            },
	{ CL_INVALID_GL_OBJECT,			        "Invalid GL Object"			            },
	{ CL_INVALID_BUFFER_SIZE,		        "Invalid Buffer Size"			        },
	{ CL_INVALID_MIP_LEVEL,			        "Invalid MIP Level"			            },
	{ CL_INVALID_GLOBAL_WORK_SIZE,		    "Invalid Global Work Size"		        },
};


const char *
PrintCLError(cl_int status)
{
	const int numErrorCodes = sizeof( ErrorCodes ) / sizeof( struct errorcode );
	for( int i = 0; i < numErrorCodes; i++ )
	{
		if( status == ErrorCodes[i].statusCode )
		{
			return ErrorCodes[i].meaning;
		}
	}

    std::stringstream ss;
    ss << "Unknown error (" << status << ")";

    return ss.str().c_str();
}
