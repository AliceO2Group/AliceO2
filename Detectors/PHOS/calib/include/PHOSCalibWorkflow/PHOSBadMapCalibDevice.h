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

#ifndef O2_CALIBRATION_PHOSBADMAP_CALIBRATOR_H
#define O2_CALIBRATION_PHOSBADMAP_CALIBRATOR_H

/// @file   PedestalCalibSpec.h
/// @brief  Device to calculate PHOS bad map

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "PHOSBase/Mapping.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "TH1.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSBadMapCalibDevice : public o2::framework::Task
{

 public:
  explicit PHOSBadMapCalibDevice(int mode) : mMode(mode){};

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);
  void calculateLimits(TH1F* h, float& mean, float& rms);
  bool calculateBadMap();
  void checkBadMap();

 private:
  int mMode = 0;                                                   /// operation mode: 0: occupancy, 1: chi2
  int mElowMin = 100;                                              /// Low E minimum in ADC counts
  int mElowMax = 200;                                              /// Low E maximum in ADC counts
  int mEhighMin = 400;                                             /// high E minimum in ADC counts
  int mEhighMax = 900;                                             /// high E maximum in ADC counts
  int mStatistics = 1000;                                          /// number of events to calculate pedestals
  static constexpr short kMinorChange = 10;                        /// ignore if number of channels changed smaller than...
  long mRunStartTime = 0;                                          /// start time of the run (milisec)
  long mValidityTime = 0;                                          /// end of validity range (milisec)
  std::unique_ptr<BadChannelsMap> mBadMap;                         //! Final calibration object
  const BadChannelsMap* mOldBadMap = nullptr;                      //! Old (current) map for comparison
  std::unique_ptr<TH1F> mMeanLow;                                  //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mMeanHigh;                                 //! Mean occupancy for high E range
  std::unique_ptr<TH1F> mMeanTime;                                 //! Mean time
  std::unique_ptr<TH1F> mChi2;                                     //! chi2 summary
  std::unique_ptr<TH1F> mChi2norm;                                 //! chi2 normalization
  std::unique_ptr<TH1F> mHGNorm;                                   //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mLGNorm;                                   //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mHGMean;                                   //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mLGMean;                                   //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mHGRMS;                                    //! Mean occupancy for low E range
  std::unique_ptr<TH1F> mLGRMS;                                    //! Mean occupancy for low E range
  std::array<short, o2::phos::Mapping::NCHANNELS + 1> mBadMapDiff; //! BadMap variation wrt previous map
};

o2::framework::DataProcessorSpec getBadMapCalibSpec(int mode);
} // namespace phos
} // namespace o2

#endif
