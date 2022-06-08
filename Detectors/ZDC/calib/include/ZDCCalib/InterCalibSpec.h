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

/// @file   InterCalibSpec.h
/// @brief  ZDC intercalibration
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_INTERCALIB_SPEC
#define O2_ZDC_INTERCALIB_SPEC

#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/Task.h"
#include "CommonDataFormat/FlatHisto1D.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "CommonUtils/NameConf.h"
#include "ZDCCalib/InterCalibData.h"
#include "ZDCCalib/InterCalib.h"
#include "ZDCCalib/InterCalibConfig.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2
{
namespace zdc
{

class InterCalibSpec : public o2::framework::Task
{
 public:
  InterCalibSpec();
  InterCalibSpec(const int verbosity);
  ~InterCalibSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void sendOutput(o2::framework::DataAllocator& output);

 private:
  int mVerbosity = DbgMinimal; // Verbosity level
  bool mInitialized = false;   // Connect once to CCDB during initialization
  InterCalib mWorker;          // Intercalibration object
  TStopwatch mTimer;
};

framework::DataProcessorSpec getInterCalibSpec();

} // namespace zdc
} // namespace o2

#endif
