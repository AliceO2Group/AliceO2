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
#include "EMCALCalib/GainCalibrationFactors.h"

#include <fairlogger/Logger.h>

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void GainCalibrationFactors::addGainCalibFactor(unsigned short iCell, float gainFactor)
{
  if (iCell >= mGainCalibFactors.size()) {
    throw CalibContainerIndexException(iCell);
  }
  mGainCalibFactors[iCell] = gainFactor;
}

float GainCalibrationFactors::getGainCalibFactors(unsigned short iCell) const
{
  if (iCell >= mGainCalibFactors.size()) {
    throw CalibContainerIndexException(iCell);
  }
  return mGainCalibFactors[iCell];
}

TH1* GainCalibrationFactors::getHistogramRepresentation() const
{
  auto hist = new TH1F("GainCalibrationFactors", "GainCalibrationFactors", 17664, 0, 17664);
  hist->SetDirectory(nullptr);

  for (std::size_t icell{0}; icell < mGainCalibFactors.size(); ++icell) {
    hist->SetBinContent(icell + 1, mGainCalibFactors[icell]);
  }

  return hist;
}

bool GainCalibrationFactors::operator==(const GainCalibrationFactors& other) const
{
  return mGainCalibFactors == other.mGainCalibFactors;
}
