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

/// \file LHCphaseCalibrationObject.cxx
/// \brief Class to store the output of the matching to TOF for calibration

#include <algorithm>
#include <cstdio>
#include "FT0Calibration/LHCphaseCalibrationObject.h"
#include "FT0Calibration/LHCClockDataHisto.h"

using namespace o2::ft0;
LHCphaseCalibrationObject LHCphaseCalibrationObjectAlgorithm::generateCalibrationObject(const LHCClockDataHisto& container)
{
  LHCphaseCalibrationObject calibrationObject;

  calibrationObject.mLHCphase = container.getGaus();

  return calibrationObject;
}
