// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDArtcPre.h
/// \author David Rohr

//#include <cub/block/block_scan.cuh>
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>

#define GPUCA_CONSMEM (gGPUConstantMemBuffer.v)
using uint64_t = unsigned long;
using uint32_t = unsigned int;
using uint16_t = unsigned short;
using uint8_t = unsigned char;
using uint = unsigned int;
using ushort = unsigned short;
using ulong = unsigned long;
#undef assert
#define assert(...)
void printf(...)
{
}
