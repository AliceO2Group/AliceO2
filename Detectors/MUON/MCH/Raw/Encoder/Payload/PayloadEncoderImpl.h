// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_ENCODER_IMPL_H
#define O2_MCH_RAW_PAYLOAD_ENCODER_IMPL_H

#include "Assertions.h"
#include "GBTEncoder.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "MakeArray.h"
#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <gsl/span>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>
#include <cassert>
#include <stdexcept>

namespace o2::mch::raw
{

/// @brief (Default) implementation of Encoder
///
/// \nosubgrouping

template <typename FORMAT, typename CHARGESUM, int VERSION>
class PayloadEncoderImpl : public PayloadEncoder
{
 public:
  PayloadEncoderImpl(Solar2FeeLinkMapper solar2feelink);

  void addChannelData(DsElecId dsId, uint8_t chId, const std::vector<SampaCluster>& data) override;

  void startHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing) override;

  size_t moveToBuffer(std::vector<std::byte>& buffer) override;

 private:
  void closeHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing);
  void gbts2buffer(uint32_t orbit, uint16_t bunchCrossing);

 private:
  uint32_t mOrbit;
  uint16_t mBunchCrossing;
  std::vector<std::byte> mBuffer;
  std::map<uint16_t, std::unique_ptr<GBTEncoder<FORMAT, CHARGESUM, VERSION>>> mGBTs;
  bool mFirstHBFrame;
  Solar2FeeLinkMapper mSolar2FeeLink;
};

template <typename FORMAT, typename CHARGESUM, int VERSION>
PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::PayloadEncoderImpl(Solar2FeeLinkMapper solar2feelink)
  : mOrbit{},
    mBunchCrossing{},
    mBuffer{},
    mGBTs{},
    mFirstHBFrame{true},
    mSolar2FeeLink{solar2feelink}
{
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::addChannelData(DsElecId dsId, uint8_t chId, const std::vector<SampaCluster>& data)
{
  auto solarId = dsId.solarId();
  auto gbt = mGBTs.find(solarId);
  if (gbt == mGBTs.end()) {
    auto f = mSolar2FeeLink(solarId);
    if (!f.has_value()) {
      throw std::invalid_argument(fmt::format("Could not get fee,link for solarId={}\n", solarId));
    }
    mGBTs.emplace(solarId, std::make_unique<GBTEncoder<FORMAT, CHARGESUM, VERSION>>(f->linkId()));
    gbt = mGBTs.find(solarId);
  }
  gbt->second->addChannelData(dsId.elinkGroupId(), dsId.elinkIndexInGroup(), chId, data);
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::gbts2buffer(uint32_t orbit, uint16_t bunchCrossing)
{
  // append to our own buffer all the words buffers from all our gbts,
  // prepending each one with a corresponding payload header

  for (auto& p : mGBTs) {
    auto& gbt = p.second;
    std::vector<std::byte> gbtBuffer;
    gbt->moveToBuffer(gbtBuffer);
    if (gbtBuffer.empty()) {
      continue;
    }
    assert(gbtBuffer.size() % 4 == 0);
    DataBlockHeader header{orbit, bunchCrossing, p.first, gbtBuffer.size()};
    appendDataBlockHeader(mBuffer, header);
    mBuffer.insert(mBuffer.end(), gbtBuffer.begin(), gbtBuffer.end());
  }
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
size_t PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::moveToBuffer(std::vector<std::byte>& buffer)
{
  closeHeartbeatFrame(mOrbit, mBunchCrossing);
  buffer.insert(buffer.end(), mBuffer.begin(), mBuffer.end());
  auto s = mBuffer.size();
  mBuffer.clear();
  return s;
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::closeHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing)
{
  gbts2buffer(orbit, bunchCrossing);
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>::startHeartbeatFrame(uint32_t orbit, uint16_t bunchCrossing)
{
  impl::assertIsInRange("bunchCrossing", bunchCrossing, 0, 0xFFF);
  // build a buffer with the _previous_ (orbit,bx)
  if (!mFirstHBFrame) {
    closeHeartbeatFrame(mOrbit, mBunchCrossing);
  }
  mFirstHBFrame = false;
  // then save the (orbit,bx) for next time
  mOrbit = orbit;
  mBunchCrossing = bunchCrossing;
}

} // namespace o2::mch::raw
#endif
