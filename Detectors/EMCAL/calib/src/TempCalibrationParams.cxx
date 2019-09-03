// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/TempCalibrationParams.h"

#include "FairLogger.h"

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void TempCalibrationParams::addTempCalibParam(unsigned short cellID, float Slope, float ParamA0)
{
  mTempCalibParamsSlope[cellID] = Slope;
  mTempCalibParamsA0[cellID] = ParamA0;
}

float TempCalibrationParams::getTempCalibParamSlope(unsigned short cellID) const
{
  return mTempCalibParamsSlope[cellID];
}

float TempCalibrationParams::getTempCalibParamA0(unsigned short cellID) const
{
  return mTempCalibParamsA0[cellID];
}

TH1* TempCalibrationParams::getHistogramRepresentationSlope() const
{

  auto hist = new TH1F("TempCalibrationParamsSlope", "Temp Calibration Params Slope", 17664, 0, 17664);
  hist->SetDirectory(nullptr);

  for (std::size_t icell{0}; icell < mTempCalibParamsSlope.size(); ++icell)
    hist->SetBinContent(icell + 1, mTempCalibParamsSlope[icell]);

  return hist;
}

TH1* TempCalibrationParams::getHistogramRepresentationA0() const
{

  auto hist = new TH1F("TempCalibrationParamsA0", "Temp Calibration Params A0", 17664, 0, 17664);
  hist->SetDirectory(nullptr);

  for (std::size_t icell{0}; icell < mTempCalibParamsA0.size(); ++icell)
    hist->SetBinContent(icell + 1, mTempCalibParamsA0[icell]);

  return hist;
}

bool TempCalibrationParams::operator==(const TempCalibrationParams& other) const
{
  return mTempCalibParamsSlope == other.mTempCalibParamsSlope && mTempCalibParamsA0 == other.mTempCalibParamsA0;
}
