// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/GainCalibrationFactors.h"

#include "FairLogger.h"

#include <TH1.h>

#include <iostream>

using namespace o2::emcal;

void GainCalibrationFactors::addGainCalibFactor(unsigned short iCell, float gainFactor)
{
  mGainCalibFactors[iCell] = gainFactor;
}

float GainCalibrationFactors::getGainCalibFactors(unsigned short iCell) const
{
  return mGainCalibFactors[iCell];
}

TH1* GainCalibrationFactors::getHistogramRepresentation() const
{
  auto hist = new TH1F("GainCalibrationFactors", "GainCalibrationFactors", 17664, 0, 17664);
  hist->SetDirectory(nullptr);

  for (std::size_t icell{0}; icell < mGainCalibFactors.size(); ++icell)
    hist->SetBinContent(icell + 1, mGainCalibFactors[icell]);

  return hist;
}

bool GainCalibrationFactors::operator==(const GainCalibrationFactors& other) const
{
  return mGainCalibFactors == other.mGainCalibFactors;
}
