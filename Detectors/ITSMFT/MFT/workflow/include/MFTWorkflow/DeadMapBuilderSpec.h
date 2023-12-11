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

/// @file   DeadMapBuilderSpec.h

#ifndef O2_MFT_DEADMAP_BUILDER_
#define O2_MFT_DEADMAP_BUILDER_

#include <sys/stat.h>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <set>
#include <deque>

#include <iostream>
#include <fstream>
#include <sstream>

// Boost library for easy access of host name
#include <boost/asio/ip/host_name.hpp>

#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/DataTakingContext.h"
#include "Framework/TimingInfo.h"
#include <fairmq/Device.h>

#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "DetectorsBase/GRPGeomHelper.h" //nicolo
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"
#include "DataFormatsDCS/DCSConfigObject.h"

// ROOT includes
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TFile.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace mft
{

class MFTDeadMapBuilder : public Task
{
 public:
  MFTDeadMapBuilder(std::string datasource);
  ~MFTDeadMapBuilder() override;

  using ChipPixelData = o2::itsmft::ChipPixelData;
  o2::itsmft::ChipMappingMFT mp;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  void stop() final;

  //////////////////////////////////////////////////////////////////
 private:
  std::string mSelfName;

  bool DebugMode = false;
  bool mDoLocalOutput = false;
  short int N_CHIPS = o2::itsmft::ChipMappingMFT::getNChips();
  int mTFLength = 32; // TODO find utility for proper value -- o2::base::GRPGeomHelper::getNHBFPerTF() returns 128 see https://github.com/AliceO2Group/AliceO2/blob/051b56f9f136e7977e83f5d26d922db9bd6ecef5/Detectors/Base/src/GRPGeomHelper.cxx#L233 and correct also default option is getSpec
  std::set<short int> mLanesAlive;
  uint mStepCounter = 0;
  uint mTFCounter = 0;

  std::string mObjectName = "mft_time_deadmap.root";
  std::string mLocalOutputDir;

  std::vector<short int>* mDeadMapTF = new std::vector<short int>{};

  Long64_t mFirstOrbitTF = 0x0;

  std::string mDataSource = "chipsstatus";

  int mTFSampling = 1000;

  TTree* mTreeObject = new TTree("map", "map");

  /// Only for debug mode
  TH2F* Htime = new TH2F("time", "time", 1000, 0, 5000, 10000, 0, 10000);

  void finalizeOutput();
  void PrepareOutputCcdb(DataAllocator& output);

  o2::framework::DataTakingContext mDataTakingContext{};
  o2::framework::TimingInfo mTimingInfo{};

  // Flag to avoid that endOfStream and stop are both done
  bool isEnded = false;

  // Run stop requested flag for EoS operations
  bool mRunStopRequested = false;
};

// Create a processor spec
o2::framework::DataProcessorSpec getMFTDeadMapBuilderSpec(std::string datasource);

} // namespace mft
} // namespace o2

#endif
