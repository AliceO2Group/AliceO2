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

#include "DataFormatsMCH/DsChannelGroup.h"
#include "DataFormatsMCH/DsChannelId.h"
#include "MCHCalibration/PedestalData.h"
#include "MCHCalibration/PedestalDigit.h"
#include "fairlogger/Logger.h"
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>

namespace o2::mch::calibration
{

void PedestalData::reset()
{
  mPedestals.clear();
}

PedestalData::PedestalMatrix initPedestalMatrix(uint16_t solarId)
{
  PedestalData::PedestalMatrix m;
  for (uint8_t c = 0; c < PedestalData::MAXCHANNEL; c++) {
    for (uint8_t d = 0; d < PedestalData::MAXDS; d++) {
      m[d][c].dsChannelId = DsChannelId{solarId, d, c};
    }
  }
  return m;
}

void PedestalData::fill(gsl::span<const PedestalDigit> digits)
{
  bool mDebug = false;

  for (auto& d : digits) {
    uint16_t solarId = d.getSolarId();
    uint8_t dsId = d.getDsId();
    uint8_t channel = d.getChannel();

    auto iPedestal = mPedestals.find(solarId);

    if (iPedestal == mPedestals.end()) {
      auto iPedestalsNew = mPedestals.emplace(std::make_pair(solarId, initPedestalMatrix(solarId)));
      iPedestal = iPedestalsNew.first;
    }

    if (iPedestal == mPedestals.end()) {
      std::cout << "[PedestalData::process] failed to insert new element\n";
      break;
    }

    auto& ped = iPedestal->second[dsId][channel];

    ped.dsChannelId = DsChannelId{solarId, dsId, channel};

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

      ped.mPedRms = ped.getRms();    // FIXME: remove this line when QC no longer uses a direct access to mPedRms
      ped.mPedMean = ped.mPedestal;  // FIXME: remove this line when QC no longer uses a direct access to mPedRms
      ped.mDsChId = ped.dsChannelId; // FIXME: remove this line when QC no longer uses a direct access to mPedRms
    }

    if (mDebug) {
      std::cout << "solarId " << (int)solarId << "  dsId " << (int)dsId << "  ch " << (int)channel << "  nsamples " << d.nofSamples()
                << "  entries " << ped.mEntries << "  ped " << ped.mPedestal << "  mVariance " << ped.mVariance << std::endl;
    }
  }
}

void PedestalData::process(const gsl::span<const o2::mch::calibration::PedestalDigit> digits)
{
  fill(digits);
}

void PedestalData::merge(const PedestalData* prev)
{
  // merge data of 2 slots
  LOGP(error, "not yet implemented");
}

void PedestalData::print() const
{
  for (const auto& p : const_cast<PedestalData&>(*this)) {
    LOGP(info, p.asString());
  }
}

PedestalData::iterator PedestalData::begin()
{
  return PedestalData::iterator(this);
}

PedestalData::iterator PedestalData::end()
{
  return PedestalData::iterator(nullptr);
}

PedestalData::const_iterator PedestalData::cbegin() const
{
  return PedestalData::const_iterator(const_cast<PedestalData*>(this));
}
PedestalData::const_iterator PedestalData::cend() const
{
  return PedestalData::const_iterator(nullptr);
}

} // namespace o2::mch::calibration
