// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUOutputControl.h
/// \author David Rohr

#ifndef GPUOUTPUTCONTROL_H
#define GPUOUTPUTCONTROL_H

#ifndef GPUCA_GPUCODE
#include <cstddef>
#endif

struct GPUOutputControl
{
	enum OutputTypeStruct {AllocateInternal = 0, UseExternalBuffer = 1, ControlledExternal = 2};
#ifndef GPUCA_GPUCODE_DEVICE
	GPUOutputControl() : OutputPtr(nullptr), Offset(0), OutputMaxSize(0), OutputType(AllocateInternal), EndOfSpace(0) {}
#endif

	const char* OutputPtr;				//Pointer to Output Space
	volatile size_t Offset;				//Offset to write into output pointer
	size_t OutputMaxSize;				//Max Size of Output Data if Pointer to output space is given
	OutputTypeStruct OutputType;		//How to perform the output
	char EndOfSpace;					// end of space flag
};

#endif
