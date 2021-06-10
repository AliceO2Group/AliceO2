// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_GBT_ENCODER_H
#define O2_MCH_RAW_ENCODER_GBT_ENCODER_H

#include <array>
#include <vector>
#include <cstdlib>
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/DataFormats.h"
#include <functional>
#include <fmt/printf.h>
#include <stdexcept>
#include "MakeArray.h"
#include "Assertions.h"
#include "MoveBuffer.h"
#include "ElinkEncoder.h"
#include "ElinkEncoderMerger.h"
#include <gsl/span>
#include "DumpBuffer.h"

namespace o2::mch::raw
{

/// @brief A GBTEncoder manages 40 ElinkEncoder to encode the data of one GBT.
///
/// Channel data is added using the addChannelData() method.
/// The encoded data (in the form of 64 bits words)
/// is exported to 8-bits words buffer using the moveToBuffer() method.
///
/// \nosubgrouping

template <typename FORMAT, typename CHARGESUM, int VERSION = 0>
class GBTEncoder
{
 public:
  /// Constructor.
  /// \param linkId of this GBT (0..11)
  GBTEncoder(uint16_t linkId);

  /** @name Main interface.
    */
  ///@{
  /// add data for one channel.
  ///
  /// \param elinkGroupId 0..7
  /// \param elinkIndexInGroup 0..4
  /// \param chId 0..63 dualSampa channel
  /// \param data vector of SampaCluster objects
  void addChannelData(uint8_t elinkGroupId, uint8_t elinkIndexInGroup, uint8_t chId, const std::vector<SampaCluster>& data);

  /// add a Heartbeat (HB) packet for a given dual sampa
  /// \param elinkGroupId 0..7
  /// \param elinkIndexInGroup 0..4
  /// \param bunchcrossing local (to sampa) bunch crossing
  void addHeartbeat(uint8_t elinkGroupId, uint8_t elinkIndexInGroup, uint20_t bunchCrossing);

  /// reset local bunch-crossing counter.
  ///
  /// (the one that is used in the sampa headers)
  void resetLocalBunchCrossing();

  /// Export our encoded data.
  ///
  /// The internal GBT words that have been accumulated so far are
  /// _moved_ (i.e. deleted from this object) to the external buffer of bytes.
  /// Returns the number of bytes added to buffer
  size_t moveToBuffer(std::vector<std::byte>& buffer);
  ///@}

  /** @name Methods for testing.
    */
  ///@{
  /// Sets to true to bypass simulation of time misalignment of elinks.
  static bool forceNoPhase;
  ///@}

  /// returns the GBT id
  uint16_t linkId() const { return mLinkId; }

  /// return the number of bytes our current buffer is
  size_t size() const { return mGbtWords.size(); }

 private:
  uint16_t mLinkId;                                                 //< id of this GBT (0..11)
  std::array<ElinkEncoder<FORMAT, CHARGESUM, VERSION>, 40> mElinks; //< the 40 Elinks we manage
  std::vector<uint64_t> mGbtWords;                                  //< the GBT words (each GBT word of 80 bits is represented by 2 64 bits words) we've accumulated so far
  ElinkEncoderMerger<FORMAT, CHARGESUM, VERSION> mElinkMerger;
};

inline int phase(int i, bool forceNoPhase)
{
  // generate the phase for the i-th ElinkEncoder
  // the default value of -1 means it will be random and decided
  // by the ElinkEncoder ctor
  //
  // if > 0 it will set a fixed phase at the beginning of the life
  // of the ElinkEncoder
  //
  // returning zero will simply disable the phase

  if (forceNoPhase) {
    return 0;
  }
  return -1;
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
bool GBTEncoder<FORMAT, CHARGESUM, VERSION>::forceNoPhase = false;

template <typename FORMAT, typename CHARGESUM, int VERSION>
GBTEncoder<FORMAT, CHARGESUM, VERSION>::GBTEncoder(uint16_t linkId)
  : mLinkId(linkId),
    mElinks{impl::makeArray<40>([](size_t i) { return ElinkEncoder<FORMAT, CHARGESUM, VERSION>(i, phase(i, GBTEncoder<FORMAT, CHARGESUM, VERSION>::forceNoPhase)); })},
    mGbtWords{},
    mElinkMerger{}
{
  impl::assertIsInRange("linkId", linkId, 0, 11);
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void GBTEncoder<FORMAT, CHARGESUM, VERSION>::addChannelData(uint8_t elinkGroupId, uint8_t elinkIndexInGroup, uint8_t chId,
                                                            const std::vector<SampaCluster>& data)
{

  impl::assertIsInRange("elinkGroupId", elinkGroupId, 0, 7);
  impl::assertIsInRange("elinkIndexInGroup", elinkIndexInGroup, 0, 4);
  mElinks.at(elinkGroupId * 5 + elinkIndexInGroup).addChannelData(chId, data);
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
void GBTEncoder<FORMAT, CHARGESUM, VERSION>::addHeartbeat(uint8_t elinkGroupId, uint8_t elinkIndexInGroup, uint20_t bunchCrossing)
{

  impl::assertIsInRange("elinkGroupId", elinkGroupId, 0, 7);
  impl::assertIsInRange("elinkIndexInGroup", elinkIndexInGroup, 0, 4);
  mElinks.at(elinkGroupId * 5 + elinkIndexInGroup).addHeartbeat(bunchCrossing);
}

template <typename FORMAT, typename CHARGESUM, int VERSION>
size_t GBTEncoder<FORMAT, CHARGESUM, VERSION>::moveToBuffer(std::vector<std::byte>& buffer)
{
  auto s = gsl::span(mElinks.begin(), mElinks.end());
  mElinkMerger(mLinkId, s, mGbtWords);
  for (auto& elink : mElinks) {
    elink.clear();
  }
  return impl::moveBuffer(mGbtWords, buffer);
}

} // namespace o2::mch::raw

#endif
