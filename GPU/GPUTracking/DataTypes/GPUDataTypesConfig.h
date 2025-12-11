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

/// \file GPUDataTypesConfig.h
/// \author David Rohr

#ifndef GPUDATATYPESCONFIG_H
#define GPUDATATYPESCONFIG_H

#include "GPUCommonDef.h"

// These are basic and non-complex data types, which will also be visible on the GPU.
// Please add complex data types required on the host but not GPU to GPUHostDataTypes.h and forward-declare!
#ifndef GPUCA_GPUCODE_DEVICE
#include <type_traits> // for bitfield below
#include <cstddef>
#endif

namespace o2::gpu
{
#include "utils/bitfield.h"

namespace gpudatatypes
{
// clang-format off
enum class GeometryType : uint32_t { RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2 };
enum DeviceType : uint32_t { INVALID_DEVICE = 0, CPU = 1, CUDA = 2, HIP = 3, OCL = 4 };
enum class GeneralStep : uint32_t { Prepare = 1, QA = 2 };
// clang-format on

enum class RecoStep : uint32_t { TPCConversion = 1,
                                 TPCSectorTracking = 2,
                                 TPCMerging = 4,
                                 TPCCompression = 8,
                                 TRDTracking = 16,
                                 ITSTracking = 32,
                                 TPCdEdx = 64,
                                 TPCClusterFinding = 128,
                                 TPCDecompression = 256,
                                 Refit = 512,
                                 AllRecoSteps = 0x7FFFFFFF,
                                 NoRecoStep = 0 };
enum class InOutType : uint32_t { TPCClusters = 1,
                                  OBSOLETE = 2,
                                  TPCMergedTracks = 4,
                                  TPCCompressedClusters = 8,
                                  TRDTracklets = 16,
                                  TRDTracks = 32,
                                  TPCRaw = 64,
                                  ITSClusters = 128,
                                  ITSTracks = 256 };
#ifndef __OPENCL__
static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL"};
static constexpr const char* const RECO_STEP_NAMES[] = {"TPC Transformation", "TPC Sector Tracking", "TPC Track Merging and Fit", "TPC Compression", "TRD Tracking", "ITS Tracking", "TPC dEdx Computation", "TPC Cluster Finding", "TPC Decompression", "Global Refit"};
static constexpr const char* const GENERAL_STEP_NAMES[] = {"Prepare", "QA"};
constexpr static int32_t N_RECO_STEPS = sizeof(gpudatatypes::RECO_STEP_NAMES) / sizeof(gpudatatypes::RECO_STEP_NAMES[0]);
constexpr static int32_t N_GENERAL_STEPS = sizeof(gpudatatypes::GENERAL_STEP_NAMES) / sizeof(gpudatatypes::GENERAL_STEP_NAMES[0]);
#endif
typedef bitfield<RecoStep, uint32_t> RecoStepField;
typedef bitfield<InOutType, uint32_t> InOutTypeField;
DeviceType GetDeviceType(const char* type);
} // namespace gpudatatypes

struct GPURecoStepConfiguration {
  gpudatatypes::RecoStepField steps = 0;
  gpudatatypes::RecoStepField stepsGPUMask = gpudatatypes::RecoStep::AllRecoSteps;
  gpudatatypes::InOutTypeField inputs = 0;
  gpudatatypes::InOutTypeField outputs = 0;
};
} // namespace o2::gpu

#endif
