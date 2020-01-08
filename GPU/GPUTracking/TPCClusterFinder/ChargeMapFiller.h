// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChargeMapFiller.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CHARGE_MAP_FILLER_H
#define O2_GPU_CHARGE_MAP_FILLER_H

#include "clusterFinderDefs.h"
#include "GPUTPCClusterFinderKernels.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class ChargeMapFiller
{

 public:
  static GPUd() void fillChargeMapImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const deprecated::Digit*, GPUglobalref() PackedCharge* chargeMap, size_t);

  static GPUd() void resetMapsImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const deprecated::Digit*, GPUglobalref() PackedCharge*, GPUglobalref() uchar*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
