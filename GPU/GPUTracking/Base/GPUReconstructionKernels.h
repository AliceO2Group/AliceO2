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

/// \file GPUReconstructionKernels.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONKERNELS_H
#define GPURECONSTRUCTIONKERNELS_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
class GPUReconstructionKernels : public T
{
 public:
  using krnlSetup = GPUReconstruction::krnlSetup;
  GPUReconstructionKernels(const GPUSettingsDeviceBackend& cfg) : T(cfg) {}

 protected:
#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward)                                                                           \
  virtual int runKernelImpl(GPUReconstruction::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, krnlSetup& _xyz GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                                       \
    return T::template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(_xyz GPUCA_M_STRIP(x_forward));                                   \
  }                                                                                                                                       \
  virtual GPUReconstruction::krnlProperties getKernelPropertiesImpl(GPUReconstruction::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>)     \
  {                                                                                                                                       \
    return T::template getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>();                                                      \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
