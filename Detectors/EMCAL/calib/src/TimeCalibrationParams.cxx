// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/TimeCalibrationParams.h"

#include "FairLogger.h"

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void TimeCalibrationParams::addTimeCalibParam(unsigned short cellID, unsigned short time, bool isLowGain)
{
  if (!isLowGain)
    mTimeCalibParamsHG[cellID] = time;
  else
    mTimeCalibParamsLG[cellID] = time;
}

void TimeCalibrationParams::addTimeCalibParamL1Phase(unsigned short iSM, unsigned char L1Phase)
{
  mTimeCalibParamsL1Phase[iSM] = L1Phase;
}

unsigned short TimeCalibrationParams::getTimeCalibParam(unsigned short cellID, bool isLowGain) const
{
  if (isLowGain)
    return mTimeCalibParamsLG[cellID];
  else
    return mTimeCalibParamsHG[cellID];
}

unsigned char TimeCalibrationParams::getTimeCalibParamL1Phase(unsigned short iSM) const
{
  return mTimeCalibParamsL1Phase[iSM];
}

TH1* TimeCalibrationParams::getHistogramRepresentation(bool isLowGain) const
{

  if (!isLowGain) {
    auto hist = new TH1S("TimeCalibrationParams", "Time Calibration Params HG", 17664, 0, 17664);
    hist->SetDirectory(nullptr);

    for (std::size_t icell{ 0 }; icell < mTimeCalibParamsHG.size(); ++icell)
      hist->SetBinContent(icell + 1, mTimeCalibParamsHG[icell]);

    return hist;
  } else {
    auto hist = new TH1S("TimeCalibrationParams", "Time Calibration Params LG", 17664, 0, 17664);
    hist->SetDirectory(nullptr);

    for (std::size_t icell{ 0 }; icell < mTimeCalibParamsLG.size(); ++icell)
      hist->SetBinContent(icell + 1, mTimeCalibParamsLG[icell]);

    return hist;
  }
}

bool TimeCalibrationParams::operator==(const TimeCalibrationParams& other) const
{
  return mTimeCalibParamsHG == other.mTimeCalibParamsHG && mTimeCalibParamsLG == other.mTimeCalibParamsLG;
}
