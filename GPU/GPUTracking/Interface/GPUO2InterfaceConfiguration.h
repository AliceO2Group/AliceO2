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
#include "GPUDataTypes.h"
#include "GPUHostDataTypes.h"
#include "GPUOutputControl.h"
#include "DataFormatsTPC/Constants.h"

class TH1F;
class TH1D;
class TH2F;

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
class GPUReconstruction;
struct GPUSettingsO2;

struct GPUInterfaceQAOutputs {
  const std::vector<TH1F>* hist1;
  const std::vector<TH2F>* hist2;
  const std::vector<TH1D>* hist3;
};

struct GPUInterfaceOutputs : public GPUTrackingOutputs {
  GPUInterfaceQAOutputs qa;
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
  GPUSettingsDisplay configDisplay;
  GPUSettingsQA configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  GPUCalibObjects configCalib;

  GPUSettingsO2 ReadConfigurableParam();

 private:
  friend class GPUReconstruction;
  GPUSettingsO2 ReadConfigurableParam_internal();
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
  const std::array<const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>*, o2::tpc::constants::MAXSECTOR>* o2DigitsMC = nullptr;
  const o2::gpu::GPUTrackingInOutZS* tpcZS = nullptr;

  // Input / Output for Merged TPC tracks, two ptrs, for the tracks themselves, and for the MC labels.
  gsl::span<o2::tpc::TrackTPC> outputTracks = {nullptr, (gsl::span<o2::tpc::TrackTPC>::index_type)0};
  gsl::span<uint32_t> outputClusRefs = {nullptr, (gsl::span<uint32_t>::index_type)0};
  gsl::span<o2::MCCompLabel> outputTracksMCTruth = {nullptr, (gsl::span<o2::MCCompLabel>::index_type)0};

  // Output for entropy-reduced clusters of TPC compression
  const o2::tpc::CompressedClustersFlat* compressedClusters = nullptr;
};
} // namespace gpu
} // namespace o2

#endif
