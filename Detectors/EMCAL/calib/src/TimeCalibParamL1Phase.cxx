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

#include "EMCALCalib/TimeCalibParamL1Phase.h"

#include <fairlogger/Logger.h>

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void TimeCalibParamL1Phase::addTimeCalibParamL1Phase(unsigned short iSM, unsigned char L1Phase)
{
  mTimeCalibParamsL1Phase[iSM] = L1Phase;
}

unsigned char TimeCalibParamL1Phase::getTimeCalibParamL1Phase(unsigned short iSM) const
{
  return mTimeCalibParamsL1Phase[iSM];
}

TH1* TimeCalibParamL1Phase::getHistogramRepresentation() const
{
  auto hist = new TH1C("hL1PhaseShift", "L1PhaseShift", 19, 0, 19);
  hist->SetDirectory(nullptr);

  for (std::size_t iSM{0}; iSM < mTimeCalibParamsL1Phase.size(); ++iSM) {
    hist->SetBinContent(iSM + 1, mTimeCalibParamsL1Phase[iSM] - '0');
  }

  return hist;
}

bool TimeCalibParamL1Phase::operator==(const TimeCalibParamL1Phase& other) const
{
  return mTimeCalibParamsL1Phase == other.mTimeCalibParamsL1Phase;
}
