// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUSettings.cxx
/// \author David Rohr

#include "GPUSettings.h"
#include "GPUDef.h"
#include "GPUDataTypes.h"
#include <cstring>

using namespace GPUCA_NAMESPACE::gpu;

void GPUSettingsRec::SetDefaults()
{
  HitPickUpFactor = 2.;
  NeighboursSearchArea = 3.f;
  ClusterError2CorrectionY = 1.f;
  ClusterError2CorrectionZ = 1.f;
  MinNTrackClusters = -1;
  MaxTrackQPt = 1.f / GPUCA_MIN_TRACK_PT_DEFAULT;
  NWays = 3;
  NWaysOuter = false;
  RejectMode = 5;
  GlobalTracking = true;
  SearchWindowDZDR = 0.f;
  TrackReferenceX = 1000.f;
  NonConsecutiveIDs = false;
  ForceEarlyTPCTransform = -1;
  DisableRefitAttachment = 0;
  dEdxTruncLow = 2;
  dEdxTruncHigh = 77;
  tpcRejectionMode = GPUSettings::RejectionStrategyA;
  tpcRejectQPt = 1.f / 0.05f;
  tpcCompressionModes = GPUSettings::CompressionFull;
  tpcCompressionSortOrder = GPUSettings::SortPad;
  tpcSigBitsCharge = 4;
  tpcSigBitsWidth = 3;
  tpcZSthreshold = 2;
  fwdTPCDigitsAsClusters = 0;
  bz0Pt = 60;
  dropLoopers = false;
  mergerCovSource = 2;
  mergerInterpolateErrors = 1;
  fitInProjections = -1;
  fitPropagateBzOnly = -1;
  retryRefit = 1;
  loopInterpolationInExtraPass = -1;
  mergerReadFromTrackerDirectly = 1;
  useMatLUT = false;
}

void GPUSettingsEvent::SetDefaults()
{
  solenoidBz = -5.00668;
  constBz = 0;
  homemadeEvents = 0;
  continuousMaxTimeBin = 0;
  needsClusterer = 0;
}

void GPUSettingsProcessing::SetDefaults()
{
  deviceType = GPUDataTypes::DeviceType::CPU;
  forceDeviceType = true;
  master = nullptr;
}

void GPUSettingsDeviceProcessing::SetDefaults()
{
  nThreads = 1;
  ompKernels = false;
  deviceNum = -1;
  platformNum = -1;
  globalInitMutex = false;
  gpuDeviceOnly = false;
  nDeviceHelperThreads = 2;
  debugLevel = -1;
  deviceTimers = true;
  debugMask = -1;
  comparableDebutOutput = true;
  resetTimers = 1;
  eventDisplay = nullptr;
  runQA = false;
  runCompressionStatistics = false;
  stuckProtection = 0;
  memoryAllocationStrategy = 0;
  keepAllMemory = false;
  keepDisplayMemory = false;
  nStreams = 8;
  trackletConstructorInPipeline = -1;
  trackletSelectorInPipeline = -1;
  trackletSelectorSlices = -1;
  forceMemoryPoolSize = 0;
  nTPCClustererLanes = 3;
  registerStandaloneInputMemory = false;
  tpcCompressionGatherMode = -1;
  mergerSortTracks = -1;
  runMC = false;
  memoryScalingFactor = 1.f;
  disableMemoryReuse = false;
  fullMergerOnGPU = true;
  alternateBorderSort = -1;
  delayedOutput = true;
  tpccfGatherKernel = true;
  prefetchTPCpageScan = false;
  doublePipeline = false;
}
