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
  NWays = 1;
  NWaysOuter = false;
  RejectMode = 5;
  GlobalTracking = true;
  SearchWindowDZDR = 0.f;
  TrackReferenceX = 1000.f;
  NonConsecutiveIDs = false;
  DisableRefitAttachment = 0;
  dEdxTruncLow = 2;
  dEdxTruncHigh = 77;
  tpcRejectionMode = 0;
  tpcRejectQPt = 1.f / 0.05f;
  tpcCompressionModes = 7;
  tpcSigBitsCharge = 4;
  tpcSigBitsWidth = 3;
}

void GPUSettingsEvent::SetDefaults()
{
  solenoidBz = -5.00668;
  constBz = 0;
  homemadeEvents = 0;
  continuousMaxTimeBin = 0;
}

void GPUSettingsProcessing::SetDefaults()
{
  deviceType = GPUDataTypes::DeviceType::CPU;
  forceDeviceType = true;
}

void GPUSettingsDeviceProcessing::SetDefaults()
{
  nThreads = 1;
  deviceNum = -1;
  platformNum = -1;
  globalInitMutex = false;
  gpuDeviceOnly = false;
  nDeviceHelperThreads = 2;
  debugLevel = -1;
  debugMask = -1;
  comparableDebutOutput = true;
  resetTimers = 1;
  eventDisplay = nullptr;
  runQA = false;
  runCompressionStatistics = false;
  stuckProtection = 0;
  memoryAllocationStrategy = 0;
  keepAllMemory = false;
  nStreams = 8;
  trackletConstructorInPipeline = true;
  trackletSelectorInPipeline = false;
}
