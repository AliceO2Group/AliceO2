// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionHIPInternals.h
/// \author David Rohr

// All HIP-header related stuff goes here, so we can run CING over GPUReconstructionHIP
#ifndef GPURECONSTRUCTIONHIPINTERNALS_H
#define GPURECONSTRUCTIONHIPINTERNALS_H

#include <hip/hip_runtime.h>
#include "GPULogging.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUReconstructionHIPInternals {
  hipStream_t Streams[GPUCA_MAX_STREAMS]; // Pointer to array of HIP Streams
};

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

static_assert(std::is_convertible<hipEvent_t, void*>::value, "HIP event type incompatible to deviceEvent");

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
