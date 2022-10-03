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

/// \file testGPUCATracking.cxx
/// \brief This task tests the TPC CA Tracking library
/// \author David Rohr

#define BOOST_TEST_MODULE Test TPC CATracking
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#include "CorrectionMapsHelper.h"
#include "TPCFastTransform.h"
#include "GPUO2Interface.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCPadGainCalib.h"
#include "CalibdEdxContainer.h"

using namespace o2::gpu;

#include <vector>
#include <iostream>
#include <iomanip>

using namespace o2::dataformats;

namespace o2
{
namespace tpc
{

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(CATracking_test1)
{
  GPUO2Interface tracker;

  float solenoidBz = -5.00668; //B-field
  float refX = 1000.;          //transport tracks to this x after tracking, >500 for disabling
  bool continuous = false;     //time frame data v.s. triggered events

  GPUO2InterfaceConfiguration config;
  config.configDeviceBackend.deviceType = GPUDataTypes::DeviceType::CPU;
  config.configDeviceBackend.forceDeviceType = true;

  config.configProcessing.ompThreads = 4;         //4 threads if we run on the CPU, 1 = default, 0 = auto-detect
  config.configProcessing.runQA = false;          //Run QA after tracking
  config.configProcessing.eventDisplay = nullptr; //Ptr to event display backend, for running standalone OpenGL event display

  config.configGRP.solenoidBz = solenoidBz;
  config.configGRP.continuousMaxTimeBin = continuous ? GPUSettings::TPC_MAX_TF_TIME_BIN : 0; //Number of timebins in timeframe if continuous, 0 otherwise

  config.configReconstruction.tpc.nWays = 3;               //Should always be 3!
  config.configReconstruction.tpc.nWaysOuter = true;       //Will create outer param for TRD
  config.configReconstruction.tpc.searchWindowDZDR = 2.5f; //Should always be 2.5 for looper-finding and/or continuous tracking
  config.configReconstruction.tpc.trackReferenceX = refX;

  config.configWorkflow.steps.set(GPUDataTypes::RecoStep::TPCConversion, GPUDataTypes::RecoStep::TPCSliceTracking,
                                  GPUDataTypes::RecoStep::TPCMerging, GPUDataTypes::RecoStep::TPCCompression, GPUDataTypes::RecoStep::TPCdEdx);
  config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCClusters);
  config.configWorkflow.outputs.set(GPUDataTypes::InOutType::TPCMergedTracks);

  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));
  std::unique_ptr<CorrectionMapsHelper> fastTransformHelper(new CorrectionMapsHelper());
  fastTransformHelper->setCorrMap(fastTransform.get());
  config.configCalib.fastTransform = fastTransform.get();
  config.configCalib.fastTransformHelper = fastTransformHelper.get();
  auto dEdxCalibContainer = GPUO2Interface::getCalibdEdxContainerDefault();
  config.configCalib.dEdxCalibContainer = dEdxCalibContainer.get();
  std::unique_ptr<TPCPadGainCalib> gainCalib = GPUO2Interface::getPadGainCalibDefault();
  config.configCalib.tpcPadGain = gainCalib.get();

  tracker.Initialize(config);
  std::vector<ClusterNativeContainer> cont(constants::MAXGLOBALPADROW);

  for (int i = 0; i < constants::MAXGLOBALPADROW; i++) {
    cont[i].sector = 0;
    cont[i].globalPadRow = i;
    cont[i].clusters.resize(1);
    cont[i].clusters[0].setTimeFlags(2, 0);
    cont[i].clusters[0].setPad(0);
    cont[i].clusters[0].setSigmaTime(1);
    cont[i].clusters[0].setSigmaPad(1);
    cont[i].clusters[0].qMax = 10;
    cont[i].clusters[0].qTot = 50;
  }
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  std::unique_ptr<ClusterNativeAccess> clusters = ClusterNativeHelper::createClusterNativeIndex(clusterBuffer, cont, nullptr, nullptr);

  GPUTrackingInOutPointers ptrs;
  ptrs.clustersNative = clusters.get();

  int retVal = tracker.RunTracking(&ptrs);
  BOOST_CHECK_EQUAL(retVal, 0);
  BOOST_CHECK_EQUAL((int)ptrs.nMergedTracks, 1);
}
} // namespace tpc
} // namespace o2
