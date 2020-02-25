// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_USER_LOGIC_ELINK_ENCODER_H
#define O2_MCH_RAW_USER_LOGIC_ELINK_ENCODER_H

#include "ElinkEncoder.h"
#include "Assertions.h"
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MoveBuffer.h"
#include "NofBits.h"
#include <cstdlib>
#include <vector>
#include <fmt/printf.h>

namespace o2::mch::raw
{

template <typename CHARGESUM>
class ElinkEncoder<UserLogicFormat, CHARGESUM>
{
 public:
  explicit ElinkEncoder(uint8_t elinkId, int phase = 0);

  void addChannelData(uint8_t chId, const std::vector<SampaCluster>& data);

  size_t moveToBuffer(std::vector<uint64_t>& buffer, uint64_t prefix);

  void clear();

 private:
  uint8_t mElinkId; //< Elink id 0..39
  bool mHasSync;    //< whether or not we've already added a sync word
  std::vector<uint64_t> mBuffer;
  int mCurrent10BitIndex; // 0..4
};

namespace
{
uint64_t dsId64(int dsId)
{
  return (static_cast<uint64_t>(dsId & 0x3F) << 53);
}

uint64_t error64(int error)
{
  return (static_cast<uint64_t>(error & 0x7) << 50);
}
} // namespace

template <typename CHARGESUM>
ElinkEncoder<UserLogicFormat, CHARGESUM>::ElinkEncoder(uint8_t elinkId,
                                                       int phase)
  : mElinkId{elinkId},
    mHasSync{false},
    mBuffer{},
    mCurrent10BitIndex{4}
{
  impl::assertIsInRange("elinkId", elinkId, 0, 39);
}

void append(uint64_t prefix, std::vector<uint64_t>& buffer, int& index, uint64_t& word, int data)
{
  word |= static_cast<uint64_t>(data) << (index * 10);
  --index;
  if (index < 0) {
    buffer.emplace_back(prefix | word);
    index = 4;
    word = 0;
  }
}

template <typename CHARGESUM>
void ElinkEncoder<UserLogicFormat, CHARGESUM>::addChannelData(uint8_t chId,
                                                              const std::vector<SampaCluster>& data)
{
  if (data.empty()) {
    throw std::invalid_argument("cannot add empty data");
  }
  assertNotMixingClusters<CHARGESUM>(data);

  int error{0}; // FIXME: what to do with error ?

  uint64_t b9 = dsId64(mElinkId) | error64(error);

  const uint64_t sync = sampaSync().uint64();

  if (!mHasSync) {
    mBuffer.emplace_back(b9 | sync);
    mHasSync = true;
  }

  auto header = buildHeader(mElinkId, chId, data);
  mBuffer.emplace_back(b9 | header.uint64());

  mCurrent10BitIndex = 4;
  CHARGESUM chargeSum;
  uint64_t word{0};
  for (auto& cluster : data) {
    append(b9, mBuffer, mCurrent10BitIndex, word, cluster.nofSamples());
    append(b9, mBuffer, mCurrent10BitIndex, word, cluster.timestamp);
    if (chargeSum() == true) {
      append(b9, mBuffer, mCurrent10BitIndex, word, cluster.chargeSum & 0x3FF);
      append(b9, mBuffer, mCurrent10BitIndex, word, (cluster.chargeSum & 0xFFC00) >> 10);
    } else {
      for (auto& s : cluster.samples) {
        append(b9, mBuffer, mCurrent10BitIndex, word, s);
      }
    }
  }
  while (mCurrent10BitIndex != 4) {
    append(b9, mBuffer, mCurrent10BitIndex, word, 0);
  }
}

template <typename CHARGESUM>
void ElinkEncoder<UserLogicFormat, CHARGESUM>::clear()
{
  mBuffer.clear();
  mHasSync = false;
}

template <typename CHARGESUM>
size_t ElinkEncoder<UserLogicFormat, CHARGESUM>::moveToBuffer(std::vector<uint64_t>& buffer, uint64_t prefix)
{
  if (mBuffer.empty()) {
    return 0;
  }
  auto n = buffer.size();
  buffer.reserve(n + mBuffer.size());
  for (auto& b : mBuffer) {
    buffer.emplace_back(b | prefix);
  }
  clear();
  return buffer.size() - n;
}
} // namespace o2::mch::raw

#endif
