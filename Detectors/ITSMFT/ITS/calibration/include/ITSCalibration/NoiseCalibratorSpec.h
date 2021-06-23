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

#ifndef O2_ITS_NOISECALIBRATORSPEC
#define O2_ITS_NOISECALIBRATORSPEC

#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

//#define TIME_SLOT_CALIBRATION
#ifdef TIME_SLOT_CALIBRATION
#include "ITSCalibration/NoiseSlotCalibrator.h"
using CALIBRATOR = o2::its::NoiseSlotCalibrator;
#else
#include "ITSCalibration/NoiseCalibrator.h"
using CALIBRATOR = o2::its::NoiseCalibrator;
#endif

using namespace o2::framework;

namespace o2
{

namespace its
{

class NoiseCalibratorSpec : public Task
{
 public:
  NoiseCalibratorSpec() = default;
  ~NoiseCalibratorSpec() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<CALIBRATOR> mCalibrator = nullptr;
};

/// create a processor spec
/// run ITS noise calibration
DataProcessorSpec getNoiseCalibratorSpec();

} // namespace its
} // namespace o2

#endif /* O2_ITS_NOISECALIBRATORSPEC */
