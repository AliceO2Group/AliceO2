#ifndef ALIGPUCAOUTPUTCONTROL_H
#define ALIGPUCAOUTPUTCONTROL_H

#ifndef GPUCA_GPUCODE
#include <cstddef>
#endif

struct AliGPUCAOutputControl
{
	enum OutputTypeStruct {AllocateInternal = 0, UseExternalBuffer = 1, ControlledExternal = 2};
	
	AliGPUCAOutputControl() : OutputPtr(NULL), Offset(0), OutputMaxSize(0), OutputType(AllocateInternal), EndOfSpace(0) {}
	const char* OutputPtr;				//Pointer to Output Space
	volatile size_t Offset;				//Offset to write into output pointer
	size_t OutputMaxSize;				//Max Size of Output Data if Pointer to output space is given
	OutputTypeStruct OutputType;		//How to perform the output
	char EndOfSpace;					// end of space flag
};

#endif
