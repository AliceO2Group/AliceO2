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

/// \file GainCalibHistos.cxx
/// \brief Class to store the output of the global tracking based TRD calibration

#include "DataFormatsTRD/GainCalibHistos.h"
#include <fairlogger/Logger.h>
#include <algorithm>

using namespace o2::trd;
using namespace o2::trd::constants;

void GainCalibHistos::init()
{
  mdEdxEntries.resize(constants::MAXCHAMBER * constants::NBINSGAINCALIB, 0);
  mInitialized = true;
}

void GainCalibHistos::reset()
{
  std::fill(mdEdxEntries.begin(), mdEdxEntries.end(), 0);
  mNEntriesTot = 0;
}

void GainCalibHistos::addEntry(float dEdx, int chamberId)
{
  // add entry for given dEdx
  int chamberOffset = chamberId * NBINSGAINCALIB;
  int iBin = (int)dEdx;
  if (iBin < 0 || iBin >= NBINSGAINCALIB) {
    // This could happen because of local gain correction but should be very rare, so we can just skip it
    return;
  }
  ++mdEdxEntries[chamberOffset + iBin];
  ++mNEntriesTot;
}

void GainCalibHistos::fill(const std::unique_ptr<const GainCalibHistos, o2::framework::InputRecord::Deleter<const o2::trd::GainCalibHistos>>& input)
{
  if (!mInitialized) {
    init();
  }
  for (int i = 0; i < MAXCHAMBER * NBINSGAINCALIB; ++i) {
    mdEdxEntries[i] += input->getHistogramEntry(i);
    mNEntriesTot += input->getHistogramEntry(i);
  }
}

void GainCalibHistos::merge(const GainCalibHistos* prev)
{
  for (int i = 0; i < MAXCHAMBER * NBINSGAINCALIB; ++i) {
    mdEdxEntries[i] += prev->getHistogramEntry(i);
    mNEntriesTot += prev->getHistogramEntry(i);
  }
}

void GainCalibHistos::print()
{
  LOG(info) << "There are " << mNEntriesTot << " entries in the container";
}
