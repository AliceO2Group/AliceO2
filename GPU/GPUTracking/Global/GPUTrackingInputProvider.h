// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingInputProvider.h
/// \author David Rohr

#ifndef GPUTRACKINGINPUTPROVIDER_H
#define GPUTRACKINGINPUTPROVIDER_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTrackingInOutZS;

class GPUTrackingInputProvider : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersInputZS(void* mem);
  void* SetPointersInputGPUOnly(void* mem);
#endif

  unsigned short mResourceZS = -1;

  bool holdsTPCZS = false;

  GPUTrackingInOutZS* mPzsMeta = nullptr;
  unsigned int* mPzsSizes = nullptr;
  void** mPzsPtrs = nullptr;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
