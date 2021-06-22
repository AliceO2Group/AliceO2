// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "BareElinkDecoder.h"

namespace o2::mch::raw
{
template <>
void BareElinkDecoder<ChargeSumMode>::sendCluster()
{
  SampaChannelHandler handler = mDecodedDataHandlers.sampaChannelHandler;
  if (handler) {
    handler(mDsId, getDualSampaChannelId(mSampaHeader),
            SampaCluster(mTimestamp, mSampaHeader.bunchCrossingCounter(), mClusterSum, mClusterSize));
  }
}

template <>
void BareElinkDecoder<SampleMode>::sendCluster()
{
  SampaChannelHandler handler = mDecodedDataHandlers.sampaChannelHandler;
  if (handler) {
    handler(mDsId, getDualSampaChannelId(mSampaHeader),
            SampaCluster(mTimestamp, mSampaHeader.bunchCrossingCounter(), mSamples));
  }
  mSamples.clear();
}
template <>
void BareElinkDecoder<ChargeSumMode>::changeToReadingData()
{
  changeState(State::ReadingClusterSum, 20);
}

template <>
void BareElinkDecoder<SampleMode>::changeToReadingData()
{
  changeState(State::ReadingSample, 10);
}

std::string bitBufferString(const std::bitset<50>& bs, int imax)
{
  std::string s;
  for (int i = 0; i < 64; i++) {
    if ((static_cast<uint64_t>(1) << i) > imax) {
      break;
    }
    if (bs.test(i)) {
      s += "1";
    } else {
      s += "0";
    }
  }
  return s;
}

} // namespace o2::mch::raw
