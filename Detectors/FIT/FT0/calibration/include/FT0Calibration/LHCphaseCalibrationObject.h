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

/// \file CalibLHCphaseFT0.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_LHCPHASE_CALIBRATION_OBJECT_H
#define ALICEO2_LHCPHASE_CALIBRATION_OBJECT_H

#include <vector>
#include "Rtypes.h"
#include "DataFormatsFT0/RawEventData.h"

namespace o2
{
namespace ft0
{
struct LHCphaseCalibrationObject {
  // LHCphase calibration
  int mLHCphase; ///< <LHCphase>

  ClassDefNV(LHCphaseCalibrationObject, 1);
};

class LHCClockDataHisto;
class LHCphaseCalibrationObjectAlgorithm
{
 public:
  [[nodiscard]] static LHCphaseCalibrationObject generateCalibrationObject(const o2::ft0::LHCClockDataHisto& container);
};

} // namespace ft0
} // namespace o2
#endif
