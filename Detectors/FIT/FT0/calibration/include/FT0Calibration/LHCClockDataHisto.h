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

#ifndef FT0_LHCPHASE_CALIBRATION_H_
#define FT0_LHCPHASE_CALIBRATION_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/LHCphaseCalibrationObject.h"
#include "CommonConstants/LHCConstants.h"
#include <array>
#include <boost/histogram.hpp>

namespace o2
{
namespace ft0
{

class LHCClockDataHisto final
{
  static constexpr int RANGE = 200; //* o2::constants::lhc::LHCBunchSpacingNS * 0.5; // BC in PS
  static constexpr int NBINS = 400;

 public:
  explicit LHCClockDataHisto(std::size_t minEntries)
  {
    //    mHisto.resize(NBINS, 0.);
  }

  size_t getEntries() const { return mEntries; }
  void print() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  void merge(const LHCClockDataHisto* prev);
  [[nodiscard]] bool hasEnoughEntries() const;
  int getGaus() const;

 private:
  std::size_t mMinEntries = 1000;
  std::array<int, NBINS> mHisto{};
  int mEntries = 0;

  ClassDefNV(LHCClockDataHisto, 1);
};

} // end namespace ft0
} // end namespace o2

#endif /* FT0_LHCPHASE_DATA_HISTO_H_ */
