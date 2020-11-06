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

#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <gsl/gsl>
#include "GPUSettings.h"
#include "GPUDisplayConfig.h"
#include "GPUQAConfig.h"
#include "GPUDataTypes.h"
#include "GPUHostDataTypes.h"
#include "DataFormatsTPC/Constants.h"

namespace o2
{
namespace tpc
{
class TrackTPC;
class Digit;
}
namespace gpu
{
class TPCFastTransform;
struct GPUSettingsO2;

// This defines an output region. Ptr points to a memory buffer, which should have a proper alignment.
// Since DPL does not respect the alignment of data types, we do not impose anything specic but just use a char data type, but it should be >= 64 bytes ideally.
// The size defines the maximum possible buffer size when GPUReconstruction is called, and returns the number of filled bytes when it returns.
// If ptr == nullptr, there is no region defined and GPUReconstruction will write its output to an internal buffer.
// If allocator is set, it is called as a callback to provide a ptr to the memory.
struct GPUInterfaceOutputRegion {
  void* ptr = nullptr;
  size_t size = 0;
  std::function<void*(size_t)> allocator = nullptr;
};

struct GPUInterfaceOutputs {
  GPUInterfaceOutputRegion compressedClusters;
  GPUInterfaceOutputRegion clustersNative;
  GPUInterfaceOutputRegion tpcTracks;
  GPUInterfaceOutputRegion clusterLabels;
};

// Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration {
  GPUO2InterfaceConfiguration() = default;
  ~GPUO2InterfaceConfiguration() = default;
  GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;

  // Settings for the Interface class
  struct GPUInterfaceSettings {
    int dumpEvents = 0;
    bool outputToExternalBuffers = false;
    bool dropSecondaryLegs = true;
    float memoryBufferScaleFactor = 1.f;
    // These constants affect GPU memory allocation only and do not limit the CPU processing
    unsigned long maxTPCZS = 8192ul * 1024 * 1024;
    unsigned int maxTPCHits = 1024 * 1024 * 1024;
    unsigned int maxTRDTracklets = 128 * 1024;
    unsigned int maxITSTracks = 96 * 1024;
  };

  GPUSettingsDeviceBackend configDeviceBackend;
  GPUSettingsProcessing configProcessing;
  GPUSettingsEvent configEvent;
  GPUSettingsRec configReconstruction;
  GPUDisplayConfig configDisplay;
  GPUQAConfig configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  GPUCalibObjects configCalib;

  GPUSettingsO2 ReadConfigurableParam();
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
  // Input: TPC clusters in cluster native format, or digits, or list of ZS pages -  const as it can only be input
  const o2::tpc::ClusterNativeAccess* clusters = nullptr;
  const std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::constants::MAXSECTOR>* o2Digits = nullptr;
  std::array<const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>*, o2::tpc::constants::MAXSECTOR>* o2DigitsMC = nullptr;
  const o2::gpu::GPUTrackingInOutZS* tpcZS = nullptr;

  // Input / Output for Merged TPC tracks, two ptrs, for the tracks themselves, and for the MC labels.
  std::vector<o2::tpc::TrackTPC>* outputTracks = nullptr;
  std::vector<uint32_t>* outputClusRefs = nullptr;
  std::vector<o2::MCCompLabel>* outputTracksMCTruth = nullptr;

  // Output for entropy-reduced clusters of TPC compression
  const o2::tpc::CompressedClustersFlat* compressedClusters = nullptr;

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
