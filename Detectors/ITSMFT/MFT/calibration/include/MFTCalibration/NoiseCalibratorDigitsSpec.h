// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorDigitsSpec.h

#ifndef O2_MFT_NOISECALIBRATORDIGITSSPEC
#define O2_MFT_NOISECALIBRATORDIGITSSPEC

#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "MFTCalibration/NoiseCalibratorDigits.h"
using CALIBRATORDIGITS = o2::mft::NoiseCalibratorDigits;

#include "DataFormatsITSMFT/NoiseMap.h"

using namespace o2::framework;

namespace o2
{

namespace mft
{

class NoiseCalibratorDigitsSpec : public Task
{
 public:
  NoiseCalibratorDigitsSpec() = default;
  ~NoiseCalibratorDigitsSpec() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<CALIBRATORDIGITS> mCalibrator = nullptr;
  std::string mMeta;
  double mThresh;
  int64_t mStart;
  int64_t mEnd;
};

/// create a processor spec
/// run MFT noise calibration
DataProcessorSpec getNoiseCalibratorDigitsSpec();

} // namespace mft
} // namespace o2

#endif /* O2_MFT_NOISECALIBRATORDIGITSSPEC */
