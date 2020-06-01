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
  float HitPickUpFactor;                 // multiplier for the chi2 window for hit pick up procedure
  float NeighboursSearchArea;            // area in cm for the search of neighbours
  float ClusterError2CorrectionY;        // correction for the squared cluster error during tracking
  float ClusterError2CorrectionZ;        // correction for the squared cluster error during tracking
  int MinNTrackClusters;                 //* required min number of clusters on the track
  float MaxTrackQPt;                     //* required max Q/Pt (==min Pt) of tracks
  char NWays;                            // Do N fit passes in final fit of merger
  char NWaysOuter;                       // Store outer param
  char RejectMode;                       // 0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits
  char GlobalTracking;                   // Enable Global Tracking (prolong tracks to adjacent sectors to find short segments)
  float SearchWindowDZDR;                // Use DZDR window for seeding instead of vertex window
  float TrackReferenceX;                 // Transport all tracks to this X after tracking (disabled if > 500)
  char NonConsecutiveIDs;                // Non-consecutive cluster IDs as in HLT, disables features that need access to slice data in TPC merger
  char ForceEarlyTPCTransform;           // Force early TPC transformation also for continuous data (-1 = auto)
  unsigned char DisableRefitAttachment;  // Bitmask to disable cluster attachment steps in refit: 1: attachment, 2: propagation, 4: loop following, 8: mirroring
  unsigned char dEdxTruncLow;            // Low truncation threshold, fraction of 128
  unsigned char dEdxTruncHigh;           // High truncation threshold, fraction of 128
  unsigned char tpcRejectionMode;        // 0: do not reject clusters, 1: do reject identified junk, 2: reject everything but good tracks
  float tpcRejectQPt;                    // Reject tracks below this Pt
  unsigned char tpcCompressionModes;     // Enabled steps of TPC compression as flags: 1=truncate charge/width LSB, 2=differences, 4=track-model
  unsigned char tpcCompressionSortOrder; // Sort order for clusters storred as differences (0 = time, 1 = pad, 2 = Z-curve-time-pad, 3 = Z-curve-pad-time)
  unsigned char tpcSigBitsCharge;        // Number of significant bits for TPC cluster charge in compression mode 1
  unsigned char tpcSigBitsWidth;         // Number of significant bits for TPC cluster width in compression mode 1
  unsigned char tpcZSthreshold;          // TPC Zero Suppression Threshold (for loading digits / forwarging digits as clusters)
  unsigned char fwdTPCDigitsAsClusters;  // Simply forward TPC digits as clusters
  unsigned char bz0Pt;                   // Nominal Pt to set when bz = 0 (in 10 MeV)
  unsigned char dropLoopers;             // Drop all clusters after starting from the second loop from tracks
  unsigned char mergerCovSource;         // 0 = simpleFilterErrors, 1 = use from track following
  unsigned char mergerInterpolateErrors; // Use interpolation for cluster rejection based on chi-2 instead of extrapolation
  char fitInProjections;                 // -1 for automatic
  char fitPropagateBzOnly;               // Use only Bz for the propagation during the fit in the first n passes, -1 = NWays -1
  char retryRefit;                       // Retry refit with larger cluster errors when fit fails
  char loopInterpolationInExtraPass;     // Perform the loop interpolation in an extra pass
  char mergerReadFromTrackerDirectly;    // Make the TPC merger read the output directly from the tracker class
  char useMatLUT;                        // Use material lookup table for TPC refit
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
  int needsClusterer;       // Set to true if the data requires the clusterizer
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

  unsigned int deviceType;   // Device type, shall use GPUDataTypes::DEVICE_TYPE constants, e.g. CPU / CUDA
  char forceDeviceType;      // Fail if device initialization fails, otherwise falls back to CPU
  GPUReconstruction* master; // GPUReconstruction master object
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
  bool ompKernels;                    // OMP Parallelization inside kernels
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
  bool keepDisplayMemory;             // Like keepAllMemory, but only for memory required for event display
  int nStreams;                       // Number of parallel GPU streams
  char trackletConstructorInPipeline; // Run tracklet constructor in pileline like the preceeding tasks instead of as one big block
  char trackletSelectorInPipeline;    // Run tracklet selector in pipeline, requres also tracklet constructor in pipeline
  char trackletSelectorSlices;        // Number of slices to processes in parallel at max
  size_t forceMemoryPoolSize;         // Override size of memory pool to be allocated on GPU / Host (set =1 to force allocating all device memory, if supported)
  int nTPCClustererLanes;             // Number of TPC clusterers that can run in parallel
  bool deviceTimers;                  // Use device timers instead of host-based timers
  bool registerStandaloneInputMemory; // Automatically register memory for the GPU which is used as input for the standalone benchmark
  int tpcCompressionGatherMode;       // Modes: 0 = gather by DMA, 1 = DMA + gather on host, 2 = gather by kernel
  bool mergerSortTracks;              // Sort track indices for GPU track fit
  bool runMC;                         // Process MC labels
  float memoryScalingFactor;          // Factor to apply to all memory scalers
  bool fitSlowTracksInOtherPass;      // Do a second pass on tracks that are supposed to take long, an attempt to reduce divergence on the GPU
  bool fullMergerOnGPU;               // Perform full TPC track merging on GPU instead of only refit
  bool alternateBorderSort;           // Alternative scheduling for sorting of border tracks
  bool delayedOutput;                 // Delay output to be parallel to track fit
  bool tpccfGatherKernel;             // Use a kernel instead of the DMA engine to gather the clusters
  bool prefetchTPCpageScan;           // Prefetch headers during TPC page scan
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
