// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

} // namespace o2::mch::raw
