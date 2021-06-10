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

#include "Assertions.h"
#include "ElinkEncoder.h"
#include "EncoderImplHelper.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MoveBuffer.h"
#include "NofBits.h"
#include <cstdlib>
#include <fmt/printf.h>
#include <vector>
#include "Framework/Logger.h"

namespace o2::mch::raw
{

template <typename CHARGESUM, int VERSION>
class ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>
{
 public:
  explicit ElinkEncoder(uint8_t elinkId, int phase = 0);

  void addChannelData(uint8_t chId, const std::vector<SampaCluster>& data);

  void addHeartbeat(uint20_t bunchCrossing);

  size_t moveToBuffer(std::vector<uint64_t>& buffer, uint16_t gbtId);

  void clear();

 private:
  uint8_t mElinkId; //< Elink id 0..39
  bool mHasSync;    //< whether or not we've already added a sync word
  std::vector<uint10_t> mBuffer;
};

template <typename CHARGESUM, int VERSION>
ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>::ElinkEncoder(uint8_t elinkId,
                                                                int phase)
  : mElinkId{elinkId},
    mHasSync{false},
    mBuffer{}
{
  impl::assertIsInRange("elinkId", elinkId, 0, 39);
}

template <typename CHARGESUM, int VERSION>
void ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>::addChannelData(uint8_t chId,
                                                                       const std::vector<SampaCluster>& data)
{
  if (data.empty()) {
    throw std::invalid_argument("cannot add empty data");
  }
  assertNotMixingClusters<CHARGESUM>(data);

  impl::fillUserLogicBuffer10(mBuffer, data,
                              mElinkId,
                              chId,
                              !mHasSync);

  if (!mHasSync) {
    mHasSync = true;
  }
}

template <typename CHARGESUM, int VERSION>
void ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>::addHeartbeat(uint20_t bunchCrossing)
{
  if (!mHasSync) {
    impl::appendSync(mBuffer);
    mHasSync = true;
  }
  for (auto chipAddress : std::array<uint8_t, 2>{0, 1}) {
    SampaHeader sh = sampaHeartbeat(chipAddress, bunchCrossing);
    impl::addPadding(mBuffer);
    impl::append(mBuffer, sh.uint64());
  }
}

template <typename CHARGESUM, int VERSION>
void ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>::clear()
{
  mBuffer.clear();
  mHasSync = false;
}

template <typename CHARGESUM, int VERSION>
size_t ElinkEncoder<UserLogicFormat, CHARGESUM, VERSION>::moveToBuffer(std::vector<uint64_t>& buffer, uint16_t gbtId)
{
  if (mBuffer.empty()) {
    return 0;
  }

  ULHeaderWord<VERSION> header{0};

  header.linkID = gbtId;
  header.dsID = mElinkId;
  header.incomplete = 0; // FIXME: what to do with incomplete ?
  header.error = 0;      // FIXME: what to do with error ?

  uint16_t b14 = static_cast<uint16_t>((header.word >> 50) & 0xFFFF); // keep only 14 most significant bits of header word

  auto n = buffer.size();
  impl::b10to64(mBuffer, buffer, b14);
  clear();
  return buffer.size() - n;
}
} // namespace o2::mch::raw

#endif
