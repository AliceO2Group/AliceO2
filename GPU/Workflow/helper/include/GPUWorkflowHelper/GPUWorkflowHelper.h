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

#ifndef O2_GPU_WORKFLOW_HELPER_H
#define O2_GPU_WORKFLOW_HELPER_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "GPUDataTypes.h"
#include <memory>

namespace o2::gpu
{

class GPUWorkflowHelper
{
  using GID = o2::dataformats::GlobalTrackID;

 public:
  struct tmpDataContainer;
  static std::shared_ptr<const tmpDataContainer> fillIOPtr(GPUTrackingInOutPointers& ioPtr, const o2::globaltracking::RecoContainer& recoCont, bool useMC, const GPUCalibObjectsConst* calib = nullptr, GID::mask_t maskCl = GID::MASK_ALL, GID::mask_t maskTrk = GID::MASK_ALL, GID::mask_t maskMatch = GID::MASK_ALL);
};

} // namespace o2::gpu

#endif
