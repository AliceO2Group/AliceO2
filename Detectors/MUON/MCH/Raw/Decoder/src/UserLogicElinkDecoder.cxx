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

#include "UserLogicElinkDecoder.h"

namespace o2::mch::raw
{

template <>
void UserLogicElinkDecoder<SampleMode>::prepareAndSendCluster()
{
  if (mDecodedDataHandlers.sampaChannelHandler) {
    SampaCluster sc(mClusterTime, mSampaHeader.bunchCrossingCounter(), mSamples);
    sendCluster(sc);
  }
  mSamples.clear();
}

template <>
void UserLogicElinkDecoder<ChargeSumMode>::prepareAndSendCluster()
{
  if (mSamples.size() != 2) {
    throw std::invalid_argument(fmt::format("expected sample size to be 2 but it is {}", mSamples.size()));
  }
  if (mDecodedDataHandlers.sampaChannelHandler) {
    uint32_t q = (((static_cast<uint32_t>(mSamples[1]) & 0x3FF) << 10) | (static_cast<uint32_t>(mSamples[0]) & 0x3FF));
    SampaCluster sc(mClusterTime, mSampaHeader.bunchCrossingCounter(), q, mClusterSize);
    sendCluster(sc);
  }
  mSamples.clear();
}

template <>
bool UserLogicElinkDecoder<SampleMode>::checkDataHeader()
{
  int chipAddMin = mDsId.elinkIndexInGroup() * 2;
  int chipAddMax = chipAddMin + 1;

  // the chip address from the SAMPA header must be consistent with
  // the e-Link index
  int chipAdd = mSampaHeader.chipAddress();
  if (chipAdd < chipAddMin || chipAdd > chipAddMax) {
    return false;
  }

  // we expect at least 3 10-bit words
  int nof10BitWords = mSampaHeader.nof10BitWords();
  if (nof10BitWords <= 2) {
    return false;
  }

  return true;
}

template <>
bool UserLogicElinkDecoder<ChargeSumMode>::checkDataHeader()
{
  int chipAddMin = mDsId.elinkIndexInGroup() * 2;
  int chipAddMax = chipAddMin + 1;

  // the chip address from the SAMPA header must be consistent with
  // the e-Link index
  int chipAdd = mSampaHeader.chipAddress();
  if (chipAdd < chipAddMin || chipAdd > chipAddMax) {
    return false;
  }

  // we expect at least 3 10-bit words
  int nof10BitWords = mSampaHeader.nof10BitWords();
  if (nof10BitWords <= 2) {
    return false;
  }
  // in cluster sum mode the number of 10-bit words must be a multiple of 4
  if ((nof10BitWords % 4) != 0) {
    return false;
  }

  return true;
}

} // namespace o2::mch::raw
