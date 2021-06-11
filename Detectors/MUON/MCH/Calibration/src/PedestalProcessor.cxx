// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHCalibration/PedestalProcessor.h"
#include <cmath>
#include <iostream>

namespace o2::mch::calibration
{

double PedestalProcessor::PedestalRecord::getRms()
{
  double rms = std::sqrt(mVariance / mEntries);
  return rms;
}

PedestalProcessor::PedestalProcessor()
{
  reset();
}

void PedestalProcessor::reset()
{
  mPedestals.clear();
}

void PedestalProcessor::process(gsl::span<const PedestalDigit> digits)
{
  bool mDebug = false;

  for (auto& d : digits) {
    auto solarId = d.getSolarId();
    auto dsId = d.getDsId();
    auto channel = d.getChannel();

    auto iPedestal = mPedestals.find(solarId);

    if (iPedestal == mPedestals.end()) {
      auto iPedestalsNew = mPedestals.emplace(std::make_pair(solarId, PedestalMatrix()));
      iPedestal = iPedestalsNew.first;
    }

    if (iPedestal == mPedestals.end()) {
      std::cout << "[PedestalProcessor::process] failed to insert new element\n";
      break;
    }

    auto& ped = iPedestal->second[dsId][channel];

    for (uint16_t i = 0; i < d.nofSamples(); i++) {
      auto s = d.getSample(i);

      ped.mEntries += 1;
      uint64_t N = ped.mEntries;

      double p0 = ped.mPedestal;
      double p = p0 + (s - p0) / N;
      ped.mPedestal = p;

      double M0 = ped.mVariance;
      double M = M0 + (s - p0) * (s - p);
      ped.mVariance = M;
    }

    if (mDebug) {
      std::cout << "solarId " << (int)solarId << "  dsId " << (int)dsId << "  ch " << (int)channel << "  nsamples " << d.nofSamples()
                << "  entries " << ped.mEntries << "  ped " << ped.mPedestal << "  variance " << ped.mVariance << std::endl;
    }
  }
}

} // namespace o2::mch::calibration
