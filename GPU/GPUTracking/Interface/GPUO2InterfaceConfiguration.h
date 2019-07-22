// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceConfiguration.h
/// \author David Rohr

#ifndef GPUO2INTERFACECONFIGURATION_H
#define GPUO2INTERFACECONFIGURATION_H

#ifndef GPUCA_O2_LIB
#define GPUCA_O2_LIB
#endif
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif

#include <memory>
#include "GPUSettings.h"
#include "GPUDisplayConfig.h"
#include "GPUQAConfig.h"
#include "GPUDataTypes.h"

namespace o2
{
class MCCompLabel;
namespace base
{
class MatLayerCylSet;
}
namespace trd
{
class TRDGeometryFlat;
}
namespace dataformats
{
template <class T>
class MCTruthContainer;
}
namespace tpc
{
class TrackTPC;
}
namespace gpu
{
class TPCFastTransform;
// Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration {
  GPUO2InterfaceConfiguration() = default;
  ~GPUO2InterfaceConfiguration() = default;
  GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;

  // Settings for the Interface class
  struct GPUInterfaceSettings {
    bool dumpEvents = false;
  };

  GPUSettingsProcessing configProcessing;
  GPUSettingsDeviceProcessing configDeviceProcessing;
  GPUSettingsEvent configEvent;
  GPUSettingsRec configReconstruction;
  GPUDisplayConfig configDisplay;
  GPUQAConfig configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  const TPCFastTransform* fastTransform = nullptr;
  const o2::base::MatLayerCylSet* matLUT = nullptr;
  const o2::trd::TRDGeometryFlat* trdGeometry = nullptr;
};

// Structure with pointers to actual data for input and output
// Which ptr is used for input and which for output is defined in GPUO2InterfaceConfiguration::configWorkflow
// Inputs and outputs are mutually exclusive.
// Inputs which are nullptr are considered empty, and will not throw an error.
// Outputs, which point to std::[container] / MCTruthContainer, will be filled and no output
// is written if the ptr is a nullptr.
// Outputs, which point to other structures are set by GPUCATracking to the location of the output. The previous
// value of the pointer is overridden. GPUCATracking will try to place the output in the "void* outputBuffer"
// location if it is not a nullptr.
struct GPUO2InterfaceIOPtrs {
  // TPC clusters in cluster native format, const as it can only be input
  const o2::tpc::ClusterNativeAccess* clusters = nullptr;

  // Input / Output for Merged TPC tracks, two ptrs, for the tracks themselves, and for the MC labels.
  std::vector<o2::tpc::TrackTPC>* outputTracks = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* outputTracksMCTruth = nullptr;

  // Output for entropy-reduced clusters of TPC compression
  const o2::tpc::CompressedClusters* compressedClusters;

  // Hint for GPUCATracking to place its output in this buffer if possible.
  // This enables to create the output directly in a shared memory segment of the framework.
  // This allows further processing with zero-copy.
  // So far this is only a hint, GPUCATracking will not always follow.
  // If outputBuffer = nullptr, GPUCATracking will allocate the output internally and own the memory.
  // TODO: Make this mandatory if outputBuffer != nullptr, and throw an error if outputBufferSize is too small.
  void* outputBuffer = nullptr;
  size_t outputBufferSize = 0;
};
} // namespace gpu
} // namespace o2

#endif
