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

/// @file   DCSAdaposParserSpec.h

#ifndef O2_ITS_DCS_PARSER_SPEC_H
#define O2_ITS_DCS_PARSER_SPEC_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include <fairmq/Device.h>

#include <ITSMFTReconstruction/RawPixelDecoder.h>

#include "Framework/RawDeviceService.h"
#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CcdbApi.h"

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

using DPCOM = o2::dcs::DataPointCompositeObject;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

class ITSDCSAdaposParser : public Task
{
 public:
  ITSDCSAdaposParser();
  ~ITSDCSAdaposParser() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

  //////////////////////////////////////////////////////////////////
 private:
  // Helper functions
  void process(const gsl::span<const DPCOM> dps);
  void processDP(const DPCOM& dpcom);
  void pushToCCDB(ProcessingContext&);
  void getCurrentCcdbAlpideParam();

  // Ccdb url for ccdb upload withing the wf
  std::string mCcdbUrl = "";

  // store the strobe length for each DPID = stave
  std::unordered_map<DPID, DPVAL> mDPstrobe;
  double mStrobeToUpload = 0.;
  bool doStrobeUpload = false;

  std::string mSelfName;
  bool mVerboseOutput = false;

  // for ccdb alpide param fetching
  o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>* mCcdbAlpideParam;
  std::string mCcdbFetchUrl = "http://ccdb-test.cern.ch:8080";
  o2::ccdb::BasicCCDBManager& mMgr = o2::ccdb::BasicCCDBManager::instance();
  long int startTime;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSDCSAdaposParserSpec();

} // namespace its
} // namespace o2

#endif
