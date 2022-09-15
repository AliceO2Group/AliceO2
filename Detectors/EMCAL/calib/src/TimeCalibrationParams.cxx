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

#include "EMCALCalib/CalibContainerErrors.h"
#include "EMCALCalib/TimeCalibrationParams.h"

#include "FairLogger.h"

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void TimeCalibrationParams::addTimeCalibParam(unsigned short cellID, short time, bool isLowGain)
{
  if (cellID >= mTimeCalibParamsHG.size()) {
    throw CalibContainerIndexException(cellID);
  }
  if (!isLowGain) {
    mTimeCalibParamsHG[cellID] = time;
  } else {
    mTimeCalibParamsLG[cellID] = time;
  }
}

short TimeCalibrationParams::getTimeCalibParam(unsigned short cellID, bool isLowGain) const
{
  if (cellID >= mTimeCalibParamsHG.size()) {
    throw CalibContainerIndexException(cellID);
  }
  if (isLowGain) {
    return mTimeCalibParamsLG[cellID];
  } else {
    return mTimeCalibParamsHG[cellID];
  }
}

TH1* TimeCalibrationParams::getHistogramRepresentation(bool isLowGain) const
{

  if (!isLowGain) {
    auto hist = new TH1S("TimeCalibrationParams", "Time Calibration Params HG", 17664, 0, 17664);
    hist->SetDirectory(nullptr);

    for (std::size_t icell{0}; icell < mTimeCalibParamsHG.size(); ++icell) {
      hist->SetBinContent(icell + 1, mTimeCalibParamsHG[icell]);
    }

    return hist;
  } else {
    auto hist = new TH1S("TimeCalibrationParams", "Time Calibration Params LG", 17664, 0, 17664);
    hist->SetDirectory(nullptr);

    for (std::size_t icell{0}; icell < mTimeCalibParamsLG.size(); ++icell) {
      hist->SetBinContent(icell + 1, mTimeCalibParamsLG[icell]);
    }

    return hist;
  }
}

bool TimeCalibrationParams::operator==(const TimeCalibrationParams& other) const
{
  return mTimeCalibParamsHG == other.mTimeCalibParamsHG && mTimeCalibParamsLG == other.mTimeCalibParamsLG;
}
