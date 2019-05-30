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

#include "GPUCommonMath.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackend;

class GPUSettings
{
 public:
};

// Settings concerning the reconstruction
struct GPUSettingsRec {
#ifndef GPUCA_GPUCODE
  GPUSettingsRec()
  {
    SetDefaults();
  }
  void SetDefaults();
  void SetMinTrackPt(float v) { MaxTrackQPt = CAMath::Abs(v) > 0.001 ? 1. / CAMath::Abs(v) : 1. / 0.001; }
#endif

  // There must be no bool in here, use char, as sizeof(bool) is compiler dependent and fails on GPUs!!!!!!
  float HitPickUpFactor;                // multiplier for the chi2 window for hit pick up procedure
  float NeighboursSearchArea;           // area in cm for the search of neighbours
  float ClusterError2CorrectionY;       // correction for the squared cluster error during tracking
  float ClusterError2CorrectionZ;       // correction for the squared cluster error during tracking
  int MinNTrackClusters;                //* required min number of clusters on the track
  float MaxTrackQPt;                    //* required max Q/Pt (==min Pt) of tracks
  char NWays;                           // Do N fit passes in final fit of merger
  char NWaysOuter;                      // Store outer param
  char RejectMode;                      // 0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits
  char GlobalTracking;                  // Enable Global Tracking (prolong tracks to adjacent sectors to find short segments)
  float SearchWindowDZDR;               // Use DZDR window for seeding instead of vertex window
  float TrackReferenceX;                // Transport all tracks to this X after tracking (disabled if > 500)
  char NonConsecutiveIDs;               // Non-consecutive cluster IDs as in HLT, disables features that need access to slice data in TPC merger
  unsigned char DisableRefitAttachment; // Bitmask to disable cluster attachment steps in refit: 1: attachment, 2: propagation, 4: loop following, 8: mirroring
  unsigned char dEdxTruncLow;           // Low truncation threshold, fraction of 128
  unsigned char dEdxTruncHigh;          // High truncation threshold, fraction of 128
  unsigned char tpcRejectionMode;       // 0: do not reject clusters, 1: do reject identified junk, 2: reject everything but good tracks
  float tpcRejectQPt;                   // Reject tracks below this Pt
  unsigned char tpcCompressionModes;    // Enabled steps of TPC compression as flags: 1=truncate charge/width LSB, 2=differences, 4=track-model
  unsigned char tpcSigBitsCharge;       // Number of significant bits for TPC cluster charge in compression mode 1
  unsigned char tpcSigBitsWidth;        // Number of significant bits for TPC cluster width in compression mode 1
};

// Settings describing the events / time frames
struct GPUSettingsEvent {
#ifndef GPUCA_GPUCODE
  GPUSettingsEvent()
  {
    SetDefaults();
  }
  void SetDefaults();
#endif

  // All new members must be sizeof(int)/sizeof(float) for alignment reasons!
  float solenoidBz;         // solenoid field strength
  int constBz;              // for test-MC events with constant Bz
  int homemadeEvents;       // Toy-MC events
  int continuousMaxTimeBin; // 0 for triggered events, -1 for default of 23ms
};

// Settings defining the setup of the GPUReconstruction processing (basically selecting the device / class instance)
struct GPUSettingsProcessing {
#ifndef GPUCA_GPUCODE
  GPUSettingsProcessing()
  {
    SetDefaults();
  }
  void SetDefaults();
#endif

  unsigned int deviceType; // Device type, shall use GPUDataTypes::DEVICE_TYPE constants, e.g. CPU / CUDA
  char forceDeviceType;    // Fail if device initialization fails, otherwise falls back to CPU
};

// Settings steering the processing once the device was selected
struct GPUSettingsDeviceProcessing {
#ifndef GPUCA_GPUCODE
  GPUSettingsDeviceProcessing()
  {
    SetDefaults();
  }
  void SetDefaults();
#endif

  int nThreads;                       // Numnber of threads on CPU, 0 = auto-detect
  int deviceNum;                      // Device number to use, in case the backend provides multiple devices (-1 = auto-select)
  int platformNum;                    // Platform to use, in case the backend provides multiple platforms (-1 = auto-select)
  bool globalInitMutex;               // Global mutex to synchronize initialization over multiple instances
  bool gpuDeviceOnly;                 // Use only GPU as device (i.e. no CPU for OpenCL)
  int nDeviceHelperThreads;           // Additional CPU helper-threads for CPU parts of processing with accelerator
  int debugLevel;                     // Level of debug output (-1 = silent)
  int debugMask;                      // Mask for debug output dumps to file
  bool comparableDebutOutput;         // Make CPU and GPU debug output comparable (sort / skip concurrent parts)
  int resetTimers;                    // Reset timers every event
  GPUDisplayBackend* eventDisplay;    // Run event display after processing, ptr to backend
  bool runQA;                         // Run QA after processing
  bool runCompressionStatistics;      // Run statistics and verification for cluster compression
  int stuckProtection;                // Timeout in us, When AMD GPU is stuck, just continue processing and skip tracking, do not crash or stall the chain
  int memoryAllocationStrategy;       // 0 = auto, 1 = new/delete per resource (default for CPU), 2 = big chunk single allocation (default for device)
  bool keepAllMemory;                 // Allocate all memory on both device and host, and do not reuse
  int nStreams;                       // Number of parallel GPU streams
  bool trackletConstructorInPipeline; // Run tracklet constructor in pileline like the preceeding tasks instead of as one big block
  bool trackletSelectorInPipeline;    // Run tracklet selector in pipeline, requres also tracklet constructor in pipeline
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
