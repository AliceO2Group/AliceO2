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

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include <gsl/span>

#include <fairlogger/Logger.h>
#include "FOCALReconstruction/PadDecoder.h"
#include "FOCALReconstruction/PadWord.h"

using namespace o2::focal;

void PadDecoder::reset()
{
  mData.reset();
}

void PadDecoder::decodeEvent(gsl::span<const PadGBTWord> gbtdata)
{
  LOG(debug) << "decoding pad data of size " << gbtdata.size() << "  GBT words - " << gbtdata.size() * sizeof(PadGBTWord) / sizeof(uint64_t) << " 64 bit words";
  // first 39 GBT words : ASIC data
  // Other words: Trigger data
  std::size_t asicsize = 39 * PadData::NASICS;
  auto asicwords = gbtdata.subspan(0, asicsize);
  auto triggerwords = gbtdata.subspan(asicsize, gbtdata.size() - asicsize);
  for (int iasic = 0; iasic < PadData::NASICS; iasic++) {
    // First part: ASIC words
    auto& asicdata = mData[iasic].getASIC();
    auto wordsthisAsic = asicwords.subspan(iasic * 39, 39);
    auto headerwords = wordsthisAsic[0].getASICData<ASICHeader>();
    asicdata.setFirstHeader(headerwords[0]);
    asicdata.setSecondHeader(headerwords[1]);
    int nchannels = 0;
    for (auto& datawords : wordsthisAsic.subspan(1, 36)) {
      for (auto& channelword : datawords.getASICData<ASICChannel>()) {
        asicdata.setChannel(channelword, nchannels);
        nchannels++;
      }
    }
    asicdata.setCMNs(wordsthisAsic[37].getASICData<ASICChannel>());
    asicdata.setCalibs(wordsthisAsic[38].getASICData<ASICChannel>());

    // Second part: Trigger words
    auto wordsTriggerThisAsic = triggerwords.subspan(iasic * mWin_dur, mWin_dur);
    auto& asiccont = mData[iasic];
    for (const auto trgword : wordsTriggerThisAsic) {
      asiccont.appendTriggerWord(trgword.getTriggerData());
    }
  }
}
