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

#include "EMCALCalib/TempCalibParamSM.h"

#include <fairlogger/Logger.h>

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void TempCalibParamSM::addTempCalibParamPerSM(unsigned short iSM, float ParamSM)
{
  mTempCalibParamsPerSM[iSM] = ParamSM;
}

float TempCalibParamSM::getTempCalibParamPerSM(unsigned short iSM) const
{
  return mTempCalibParamsPerSM[iSM];
}

TH1* TempCalibParamSM::getHistogramRepresentation() const
{
  auto hist = new TH1F("TempCalibParamSM", "Temp Calibration Params per SM", 19, 0, 19);
  hist->SetDirectory(nullptr);

  for (std::size_t iSM{0}; iSM < mTempCalibParamsPerSM.size(); ++iSM) {
    hist->SetBinContent(iSM + 1, mTempCalibParamsPerSM[iSM]);
  }

  return hist;
}

bool TempCalibParamSM::operator==(const TempCalibParamSM& other) const
{
  return mTempCalibParamsPerSM == other.mTempCalibParamsPerSM;
}
