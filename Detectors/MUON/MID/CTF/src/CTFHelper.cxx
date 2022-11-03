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

/// \file   CTFHelper.cxx
/// \author ruben.shahoyan@cern.ch
/// \brief  Helper for MID CTF creation

#include "MIDCTF/CTFHelper.h"
#include "CommonUtils/IRFrameSelector.h"

using namespace o2::mid;

void CTFHelper::TFData::buildReferences(o2::utils::IRFrameSelector& irSelector)
{
  uint32_t nDone = 0, idx[NEvTypes] = {};
  uint32_t sizes[NEvTypes] = {
    uint32_t(rofData[size_t(EventType::Standard)].size()),
    uint32_t(rofData[size_t(EventType::Calib)].size()),
    uint32_t(rofData[size_t(EventType::FET)].size())};
  uint64_t rofBC[NEvTypes] = {};
  auto fillNextROFBC = [&nDone, &rofBC, &idx, &sizes, this](int it) {
    if (idx[it] < sizes[it]) {
      rofBC[it] = this->rofData[it][idx[it]].interactionRecord.toLong();
    } else {
      rofBC[it] = -1;
      nDone++;
    }
  };
  for (uint32_t it = 0; it < NEvTypes; it++) {
    fillNextROFBC(it);
  }
  while (nDone < NEvTypes) { // find next ROFRecord with smallest BC, untill all 3 spans are traversed
    int selT = rofBC[0] <= rofBC[1] ? (rofBC[0] <= rofBC[2] ? 0 : 2) : (rofBC[1] <= rofBC[2] ? 1 : 2);
    if (!irSelector.isSet() || irSelector.check(rofData[selT][idx[selT]].interactionRecord) >= 0) {
      rofDataRefs.emplace_back(idx[selT], selT);
      for (uint32_t ic = rofData[selT][idx[selT]].firstEntry; ic < rofData[selT][idx[selT]].getEndIndex(); ic++) {
        colDataRefs.emplace_back(ic, selT); // register indices of corresponding column data
      }
    }
    ++idx[selT]; // increment used index
    fillNextROFBC(selT);
  }
}
