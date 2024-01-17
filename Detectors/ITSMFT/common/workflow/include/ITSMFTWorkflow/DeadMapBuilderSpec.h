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

#ifndef O2_ITSMFT_DEADMAP_BUILDER_
#define O2_ITSMFT_DEADMAP_BUILDER_

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
#include "DataFormatsITSMFT/TimeDeadMap.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"

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
namespace itsmft
{

class ITSMFTDeadMapBuilder : public Task
{
 public:
  ITSMFTDeadMapBuilder(std::string datasource, bool doMFT);
  ~ITSMFTDeadMapBuilder() override;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  void stop() final;

  //////////////////////////////////////////////////////////////////
 private:
  std::string mSelfName;

  bool mRunMFT = false;
  bool mDoLocalOutput = false;
  uint16_t N_CHIPS;
  uint16_t N_CHIPS_ITSIB = o2::itsmft::ChipMappingITS::getNChips(0);
  int mTFLength = 32; // TODO find utility for proper value -- o2::base::GRPGeomHelper::getNHBFPerTF() returns 128 see https://github.com/AliceO2Group/AliceO2/blob/051b56f9f136e7977e83f5d26d922db9bd6ecef5/Detectors/Base/src/GRPGeomHelper.cxx#L233 and correct also default option is getSpec

  uint mStepCounter = 0;
  uint mTFCounter = 0;

  std::string mObjectName;
  std::string mLocalOutputDir;

  std::string MAP_VERSION = "3"; // to change in case the encoding or the format change

  std::vector<uint16_t> mDeadMapTF{};

  unsigned long mFirstOrbitTF = 0x0;

  std::string mDataSource = "chipsstatus";

  int mTFSampling = 1000;

  o2::itsmft::TimeDeadMap mMapObject;

  void finalizeOutput();
  void PrepareOutputCcdb(DataAllocator& output);

  // Utils

  std::vector<uint16_t> getChipIDsOnSameCable(uint16_t);

  o2::framework::DataTakingContext mDataTakingContext{};
  o2::framework::TimingInfo mTimingInfo{};

  // Flag to avoid that endOfStream and stop are both done
  bool isEnded = false;

  // Run stop requested flag for EoS operations
  bool mRunStopRequested = false;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSMFTDeadMapBuilderSpec(std::string datasource, bool doMFT);

} // namespace itsmft
} // namespace o2

#endif
