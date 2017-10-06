struct _makefiles_opencl_platform_info
{
	char platform_profile[64];
	char platform_version[64];
	char platform_name[64];
	char platform_vendor[64];
	cl_uint count;
};

struct _makefiles_opencl_device_info
{
	char device_name[64];
	char device_vendor[64];
	cl_uint nbits;
	size_t binary_size;
};
