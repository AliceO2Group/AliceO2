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

/// @file   ThresholdAggregatorSpec.h

#ifndef O2_ITS_THRESHOLD_AGGREGATOR_
#define O2_ITS_THRESHOLD_AGGREGATOR_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include <fairmq/Device.h>

#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder

#include "Framework/RawDeviceService.h"
#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsDCS/DCSConfigObject.h"
#include "Framework/InputRecordWalker.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

class ITSThresholdAggregator : public Task
{
 public:
  ITSThresholdAggregator();
  ~ITSThresholdAggregator() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  void finalize(EndOfStreamContext* ec);
  void stop() final;

  //////////////////////////////////////////////////////////////////
 private:
  void finalizeOutput();
  void updateLHCPeriod(ProcessingContext&);
  void updateRunID(ProcessingContext&);

  std::unique_ptr<o2::ccdb::CcdbObjectInfo> mWrapper = nullptr;
  std::string mOutputStr;
  std::string mType;

  std::string mSelfName;

  // Helper functions for writing to the database
  bool mVerboseOutput = false;

  // Keep track of whether the endOfStream() or stop() has been called
  bool mStopped = false;

  o2::dcs::DCSconfigObject_t tuningMerge;
  o2::dcs::DCSconfigObject_t chipDoneMerge;
  short int mRunType = -1;
  // Either "T" for threshold, "V" for VCASN, or "I" for ITHR
  char mScanType = 'n';
  // Either "derivative"=0, "fit"=1, or "hitcounting=2
  char mFitType = 'n';

  std::string mLHCPeriod;
  // Ccdb url for ccdb upload withing the wf
  std::string mCcdbUrl = "";
  // Run number
  int mRunNumber = -1;
  // confDB version
  short int mDBversion = -1;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSThresholdAggregatorSpec();

} // namespace its
} // namespace o2

#endif
