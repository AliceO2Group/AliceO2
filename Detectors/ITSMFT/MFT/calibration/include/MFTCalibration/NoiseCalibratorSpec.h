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

/// @file   NoiseCalibratorSpec.h

#ifndef O2_MFT_NOISECALIBRATORSPEC
#define O2_MFT_NOISECALIBRATORSPEC

#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "MFTCalibration/NoiseCalibrator.h"
#include "DetectorsBase/GRPGeomHelper.h"
using CALIBRATOR = o2::mft::NoiseCalibrator;

//#include "MFTCalibration/NoiseSlotCalibrator.h" //For TimeSlot calibration
// using CALIBRATOR = o2::mft::NoiseSlotCalibrator;

#include "DataFormatsITSMFT/NoiseMap.h"

using namespace o2::framework;

namespace o2
{

namespace mft
{

class NoiseCalibratorSpec : public Task
{
 public:
  NoiseCalibratorSpec(bool digits = false, std::shared_ptr<o2::base::GRPGeomRequest> req = {});
  ~NoiseCalibratorSpec() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  void sendOutputCcdb(DataAllocator& output);
  void sendOutputCcdbDcs(DataAllocator& output);
  void sendOutputDcs(DataAllocator& output);
  void setOutputDcs(const o2::itsmft::NoiseMap& payload);
  o2::itsmft::NoiseMap mNoiseMap{936};
  std::unique_ptr<CALIBRATOR> mCalibrator = nullptr;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  std::string mPath;
  std::string mMeta;

  std::vector<std::array<int, 3>> mNoiseMapForDcs;
  std::string mPathDcs;
  std::string mOutputType;

  double mThresh;
  int64_t mStart;
  int64_t mEnd;
  bool mDigits = false;
  bool mStopMeOnly = false; // send QuitRequest::Me instead of QuitRequest::All
};

/// create a processor spec
/// run MFT noise calibration
DataProcessorSpec getNoiseCalibratorSpec(bool useDigits);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_NOISECALIBRATORSPEC */
