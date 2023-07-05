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

/// @file   TDCCalibSpec.h
/// @brief  ZDC TDC calibratin
/// @author luca.quaglia@cern.ch

#ifndef O2_ZDC_TDCCALIB_SPEC
#define O2_ZDC_TDCCALIB_SPEC

#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "CommonDataFormat/FlatHisto1D.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "ZDCCalib/TDCCalib.h"
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/TDCCalibConfig.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2
{
namespace zdc
{

class TDCCalibSpec : public o2::framework::Task
{
 public:
  TDCCalibSpec();
  TDCCalibSpec(const int verbosity);
  ~TDCCalibSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void sendOutput(o2::framework::EndOfStreamContext& ec);

 private:
  int mVerbosity = DbgMinimal; // Verbosity level
  bool mInitialized = false;   // Connect once to CCDB during initialization
  TDCCalib mWorker;            // TDC calibration object
  TStopwatch mTimer;
  long mRunStartTime = 0;                                                      /// start time of the run (ms)
  o2::framework::DataAllocator* mOutput = nullptr;                             /// Pointer to output object
  std::unique_ptr<o2::dataformats::FileMetaData> mHistoFileMetaData = nullptr; /// Pointer to metadata file
  std::string mOutputDir;                                                      /// where to write calibration digits
  std::string mHistoFileName;                                                  /// file name of output calib digits
  std::string mLHCPeriod;
  int mRunNumber = -1;
};

framework::DataProcessorSpec getTDCCalibSpec();

} // namespace zdc
} // namespace o2

#endif
