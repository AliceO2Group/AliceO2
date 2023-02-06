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
#include "MCHMappingInterface/Segmentation.h"
#include "fairlogger/Logger.h"
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>

namespace o2::mch::calibration
{

PedestalData::PedestalData()
{
  mSolar2FeeLinkMapper = o2::mch::raw::createSolar2FeeLinkMapper<o2::mch::raw::ElectronicMapperGenerated>();
  mElec2DetMapper = o2::mch::raw::createElec2DetMapper<o2::mch::raw::ElectronicMapperGenerated>();
}

void PedestalData::reset()
{
  mPedestals.clear();
}

PedestalData::PedestalMatrix PedestalData::initPedestalMatrix(uint16_t solarId)
{
  PedestalData::PedestalMatrix m;
  for (uint8_t d = 0; d < PedestalData::MAXDS; d++) {
    // apply the mapping from SOLAR to detector
    int deId = -1;
    int dsIddet = -1;

    o2::mch::raw::DsElecId dsElecId(solarId, d / 5, d % 5);
    std::optional<o2::mch::raw::DsDetId> dsDetId = mElec2DetMapper(dsElecId);
    if (dsDetId) {
      deId = dsDetId->deId();
      dsIddet = dsDetId->dsId();
    }

    for (uint8_t c = 0; c < PedestalData::MAXCHANNEL; c++) {

      // check if the channel is associated to a detector pad
      int padId = -1;
      if (deId >= 0 && dsIddet >= 0) {
        const o2::mch::mapping::Segmentation& segment = o2::mch::mapping::segmentation(deId);
        padId = segment.findPadByFEE(dsIddet, int(c));
      }
      if (padId >= 0) {
        m[d][c].dsChannelId = DsChannelId{solarId, d, c};
        mSize += 1;
      }
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
      LOGP(fatal, "failed to insert new element in padestals map");
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
      LOGP(info, "solarId {}  dsId {}  ch {}  nsamples {}  entries{}  mean {}  variance {}",
           (int)solarId, (int)dsId, (int)channel, d.nofSamples(), ped.mEntries, ped.mPedestal, ped.mVariance);
    }
  }
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

uint32_t PedestalData::size() const
{
  return mSize;
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
