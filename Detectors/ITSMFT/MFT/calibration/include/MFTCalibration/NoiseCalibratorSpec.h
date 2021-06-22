// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
using CALIBRATOR = o2::mft::NoiseCalibrator;

//#include "MFTCalibration/NoiseSlotCalibrator.h" //For TimeSlot calibration
//using CALIBRATOR = o2::mft::NoiseSlotCalibrator;

#include "DataFormatsITSMFT/NoiseMap.h"

using namespace o2::framework;

namespace o2
{

namespace mft
{

class NoiseCalibratorSpec : public Task
{
 public:
  NoiseCalibratorSpec(bool digits = false);
  ~NoiseCalibratorSpec() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);
  o2::itsmft::NoiseMap mNoiseMap{936};
  std::unique_ptr<CALIBRATOR> mCalibrator = nullptr;
  std::string mPath;
  std::string mMeta;
  double mThresh;
  int64_t mStart;
  int64_t mEnd;
  bool mDigits = false;
};

/// create a processor spec
/// run MFT noise calibration
DataProcessorSpec getNoiseCalibratorSpec(bool useDigits);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_NOISECALIBRATORSPEC */
