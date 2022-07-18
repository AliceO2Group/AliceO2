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

/// \file GPUHostDataTypes.h
/// \author David Rohr

#ifndef GPUHOSTDATATYPES_H
#define GPUHOSTDATATYPES_H

#include "GPUCommonDef.h"

// These are complex data types wrapped in simple structs, which can be forward declared.
// Structures used on the GPU can have pointers to these wrappers, when the wrappers are forward declared.
// These wrapped complex types are not meant for usage on GPU

#if defined(GPUCA_GPUCODE)
#error "GPUHostDataTypes.h should never be included on GPU."
#endif

#include <vector>
#include <array>
#include <atomic>
#include "DataFormatsTPC/Constants.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTPCDigitsMCInput {
  std::array<const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>*, o2::tpc::constants::MAXSECTOR> v;
};

struct GPUTPCClusterMCInterim {
  std::vector<o2::MCCompLabel> labels;
};

struct GPUTPCClusterMCInterimArray {
  std::vector<GPUTPCClusterMCInterim> data;
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
};

struct GPUTPCLinearLabels {
  std::vector<o2::dataformats::MCTruthHeaderElement> header;
  std::vector<o2::MCCompLabel> data;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
