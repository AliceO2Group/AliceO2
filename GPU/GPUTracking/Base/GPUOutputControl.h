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

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE
#include <cstddef>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUOutputControl {
  enum OutputTypeStruct { AllocateInternal = 0,
                          UseExternalBuffer = 1 };
#ifndef GPUCA_GPUCODE_DEVICE
  GPUOutputControl() = default;
#endif

  void* OutputBase = nullptr;                     // Base ptr to memory pool, occupied size is OutputPtr - OutputBase
  void* OutputPtr = nullptr;                      // Pointer to Output Space
  size_t OutputMaxSize = 0;                       // Max Size of Output Data if Pointer to output space is given
  OutputTypeStruct OutputType = AllocateInternal; // How to perform the output
  char EndOfSpace = 0;                            // end of space flag
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
