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
/// @brief  ZDC baseline calibration
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_BASELINECALIB_SPEC
#define O2_ZDC_BASELINECALIB_SPEC

#include <TStopwatch.h>
#include "Headers/DataHeader.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "ZDCCalib/BaselineCalib.h"
#include "ZDCCalib/BaselineCalibConfig.h"
#include "ZDCReconstruction/BaselineParam.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2
{
namespace zdc
{

class BaselineCalibSpec : public o2::framework::Task
{
 public:
  BaselineCalibSpec();
  BaselineCalibSpec(const int verbosity);
  ~BaselineCalibSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void sendOutput(); // Send output to CCDB
  void reset();      // Ask for a reset internal structures

 private:
  int mVerbosity = DbgMinimal; // Verbosity level
  bool mInitialized = false;   // Connect once to CCDB during initialization
  uint64_t mCTimeEnd = 0;      // Time of last processed time frame
  uint64_t mCTimeMod = 0;      // Integration time slot
  uint32_t mProcessed = 0;     // Number of cycles since last CCDB write
  BaselineCalib mWorker;       // Baseline calibration object
  TStopwatch mTimer;
  o2::framework::DataAllocator* mOutput = nullptr;                             /// Pointer to output object
  std::unique_ptr<o2::dataformats::FileMetaData> mHistoFileMetaData = nullptr; /// Pointer to metadata file
  std::string mOutputDir;                                                      /// where to write calibration digits
  std::string mHistoFileName;                                                  /// file name of output calib digits
  std::string mLHCPeriod;
  o2::header::DataHeader::RunNumberType mRunNumber = 0; /// Current run number
  long mRunStartTime = 0;                               /// start time of the run (ms)
};

framework::DataProcessorSpec getBaselineCalibSpec();

} // namespace zdc
} // namespace o2

#endif
