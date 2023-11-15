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

#ifndef O2_FT0_SPECTRAINFOOBJECT_H
#define O2_FT0_SPECTRAINFOOBJECT_H

#include <array>
#include "Rtypes.h"
#include "FT0Base/Geometry.h"

namespace o2::ft0
{

struct SpectraInfoObject {
  float mGausMean{};      // Peak of Gausian function
  float mGausRMS{};       // RMS of Gausian function
  float mGausConstant{};  // Constant of Gausian function
  float mFitChi2{};       // Chi2 of Gausian fitting
  float mStatMean{};      // Spectra mean
  float mStatRMS{};       // Spectra RMS
  float mStat{};          // Statistic
  uint32_t mStatusBits{}; // Status bits for extra info
  ClassDefNV(SpectraInfoObject, 1);
};

struct TimeSpectraInfoObject {
  std::array<SpectraInfoObject, o2::ft0::Geometry::Nchannels> mTime;
  SpectraInfoObject mTimeA;
  SpectraInfoObject mTimeC;
  SpectraInfoObject mSumTimeAC;
  SpectraInfoObject mDiffTimeCA;
  static constexpr const char* getObjectPath() { return "FT0/Calib/TimeSpectraInfo"; }
  ClassDefNV(TimeSpectraInfoObject, 1);
};

struct AmpSpectraInfoObject {
  std::array<SpectraInfoObject, o2::ft0::Geometry::Nchannels> mAmpADC0;
  std::array<SpectraInfoObject, o2::ft0::Geometry::Nchannels> mAmpADC1;
  static constexpr const char* getObjectPath() { return "FT0/Calib/AmpSpectraInfo"; }
  ClassDefNV(AmpSpectraInfoObject, 1);
};

} // namespace o2::ft0

#endif // O2_FT0_TIMESPECTRAINFOOBJECT_H
