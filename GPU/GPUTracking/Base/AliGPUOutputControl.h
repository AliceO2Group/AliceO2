#ifndef ALIGPUOUTPUTCONTROL_H
#define ALIGPUOUTPUTCONTROL_H

#ifndef GPUCA_GPUCODE
#include <cstddef>
#endif

struct AliGPUOutputControl
{
	enum OutputTypeStruct {AllocateInternal = 0, UseExternalBuffer = 1, ControlledExternal = 2};
#ifndef GPUCA_GPUCODE_DEVICE
	AliGPUOutputControl() : OutputPtr(nullptr), Offset(0), OutputMaxSize(0), OutputType(AllocateInternal), EndOfSpace(0) {}
#endif

	const char* OutputPtr;				//Pointer to Output Space
	volatile size_t Offset;				//Offset to write into output pointer
	size_t OutputMaxSize;				//Max Size of Output Data if Pointer to output space is given
	OutputTypeStruct OutputType;		//How to perform the output
	char EndOfSpace;					// end of space flag
};

#endif
