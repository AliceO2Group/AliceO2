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

/// @file   TDCCalibEPNSpec.h
/// @brief  ZDC TDC calibration pre-processing on EPN
/// @author luca.quaglia@cern.ch

#ifndef O2_ZDC_TDCCALIBEPN_SPEC
#define O2_ZDC_TDCCALIBEPN_SPEC

#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/TDCCalibEPN.h"
#include "ZDCCalib/TDCCalibConfig.h"

namespace o2
{
namespace zdc
{

class TDCCalibEPNSpec : public o2::framework::Task
{
 public:
  TDCCalibEPNSpec();
  TDCCalibEPNSpec(const int verbosity);
  ~TDCCalibEPNSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  int mVerbosity = DbgMinimal; // Verbosity level
  bool mInitialized = false;   // Connect once to CCDB during initialization
  TDCCalibEPN mWorker;         // TDC calibration object (was mTDCCalibEPN, modified after discussion with Pietro 20 June 2022)
  TStopwatch mTimer;
};

framework::DataProcessorSpec getTDCCalibEPNSpec();

} // namespace zdc
} // namespace o2

#endif