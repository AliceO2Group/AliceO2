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

#ifndef O2_CALIBRATION_PHOSHGLGRATIO_CALIBRATOR_H
#define O2_CALIBRATION_PHOSHGLGRATIO_CALIBRATOR_H

/// @file   HGLGRatioCalibSpec.h
/// @brief  Device to calculate PHOS HG/LG ratio

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "PHOSBase/Mapping.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "TH2.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSHGLGRatioCalibDevice : public o2::framework::Task
{

  union PairAmp {
    uint32_t mDataWord;
    struct {
      uint32_t mHGAmp : 16; ///< Bits  0 - 15: HG amplitude in channel
      uint32_t mLGAmp : 16; ///< Bits 16 - 25: LG amplitude in channel
    };
  };

 public:
  explicit PHOSHGLGRatioCalibDevice(bool useCCDB, bool forceUpdate) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate) {}
  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);
  void fillRatios();
  void calculateRatios();
  void checkRatios();

 private:
  bool mUseCCDB = false;
  bool mForceUpdate = false;                                      /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;                                        /// set is close to current and can update it
  static constexpr short kMinorChange = 10;                       /// ignore if number of channels changed smaller than...
  long mRunStartTime = 0;                                         /// start time of the run (sec)
  std::unique_ptr<CalibParams> mCalibParams;                      //! Final calibration object
  std::unique_ptr<const CalibParams> mOldCalibParams;             //! Previous calibration object
  int mStatistics = 100000;                                       /// number of events to calculate HG/LG ratio
  short mMinLG = 20;                                              /// minimal LG ampl used in ratio
  short minimalStatistics = 100;                                  /// minimal statistics per channel
  std::map<short, PairAmp> mMapPairs;                             //! HG/LG pair
  std::unique_ptr<TH2F> mhRatio;                                  //! Histogram with ratios
  std::array<float, o2::phos::Mapping::NCHANNELS + 1> mRatioDiff; //! Ratio variation wrt previous map
};

DataProcessorSpec getHGLGRatioCalibSpec(bool useCCDB, bool forceUpdate);

} // namespace phos
} // namespace o2

#endif
