// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUSettings.h
/// \author David Rohr

#ifndef GPUSETTINGS_H
#define GPUSETTINGS_H

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <vector>
#include <string>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackend;
class GPUReconstruction;

class GPUSettings
{
 public:
  enum CompressionModes { CompressionTruncate = 1,
                          CompressionDifferences = 2,
                          CompressionTrackModel = 4,
                          CompressionFull = 7 };
  enum CompressionSort { SortTime = 0,
                         SortPad = 1,
                         SortZTimePad = 2,
                         SortZPadTime = 3,
                         SortNoSort = 4 };
  enum CompressionRejection { RejectionNone = 0,
                              RejectionStrategyA = 1,
                              RejectionStrategyB = 2 };

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  static CONSTEXPR unsigned int TPC_MAX_TF_TIME_BIN = ((256 * 3564 + 2 * 8 - 2) / 8);
#endif
};

// Settings describing the events / time frames
struct GPUSettingsEvent {
  // All new members must be sizeof(int) resp. sizeof(float) for alignment reasons!
  GPUSettingsEvent();
  float solenoidBz;         // solenoid field strength
  int constBz;              // for test-MC events with constant Bz
  int homemadeEvents;       // Toy-MC events
  int continuousMaxTimeBin; // 0 for triggered events, -1 for default of 23ms
  int needsClusterer;       // Set to true if the data requires the clusterizer
};

// Settings defining the setup of the GPUReconstruction processing (basically selecting the device / class instance)
struct GPUSettingsDeviceBackend {
  GPUSettingsDeviceBackend();
  unsigned int deviceType;   // Device type, shall use GPUDataTypes::DEVICE_TYPE constants, e.g. CPU / CUDA
  char forceDeviceType;      // Fail if device initialization fails, otherwise falls back to CPU
  GPUReconstruction* master; // GPUReconstruction master object
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#ifdef GPUCA_GPUCODE_DEVICE
#define QCONFIG_GPU
#endif
// See GPUSettingsList.h for a list of all available settings of GPU Reconstruction
#ifndef GPUCA_GPUCODE_GENRTC
#include "utils/qconfig.h"
#endif

#endif
