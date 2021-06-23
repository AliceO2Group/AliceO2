// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_BARE_ELINK_ENCODER_H
#define O2_MCH_RAW_BARE_ELINK_ENCODER_H

#include "Assertions.h"
#include "BitSet.h"
#include "ElinkEncoder.h"
#include "EncoderImplHelper.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/SampaHeader.h"
#include "NofBits.h"
#include <fmt/format.h>
#include <fmt/printf.h>
#include <gsl/span>
#include <iostream>
#include <vector>

namespace o2::mch::raw
{

/// @brief Center piece of the MCH Bare Raw Data Format encoder logic.
///
/// Converts the data from one SampaCluster into a bit stream
/// that mimics the way Elinks see data.

template <typename CHARGESUM>
class ElinkEncoder<BareFormat, CHARGESUM>
{
 public:
  /// Constructs an Encoder for one Elink.
  ///
  /// \param elinkId is the elink identifier and _must_ be between 0 and 39
  /// \param phase can be used to simulate a different time alignment
  /// between elinks
  ///
  /// if elinkId is not within allowed range an exception is thrown.
  explicit ElinkEncoder(uint8_t elinkId, int phase = 0);

  /// addChannelData converts the SampaCluster data into a bit sequence.
  ///
  /// \param chId is the Sampa channel to associated this data with
  /// \param data is a vector of SampaCluster representing the SampaCluster(s)
  /// of this channel within one Sampa time window.
  void addChannelData(uint8_t chId, const std::vector<SampaCluster>& data);

  void addHeartbeat(uint20_t bunchCrossing);

  /// Empty the bit stream.
  void clear();

  /// fillWithSync appends bits from the sync word until the length
  /// is `upto`
  void fillWithSync(int upto);

  /// get the i-th bit in our bit stream.
  bool get(int i) const;

  /// len returns the number of bits we have in our bit stream
  int len() const;

  /// reset our local bunch crossing counter
  void resetLocalBunchCrossing();

  /// converts the bits within a range into an integer value.
  /// throws if the range [a,b] does not fit within 64 bits.
  uint64_t range(int a, int b) const;

 private:
  void append(bool value);
  void append(const SampaCluster& sc);
  void append10(uint16_t value);
  void append20(uint32_t value);
  void append50(uint64_t value);
  void appendCharges(const SampaCluster& sc);
  void assertPhase();
  void assertSync();
  uint64_t nofSync() const { return mNofSync; }

 private:
  uint8_t mElinkId;             //< Elink id 0..39
  BitSet mBitSet;               //< bitstream
  uint64_t mNofSync;            //< number of sync words seen so far
  int mSyncIndex;               //< at which sync bit position should the next fillWithSync start
  uint64_t mNofBitSeen;         //< total number of bits seen so far
  int mPhase;                   //< initial number of bits
  uint32_t mLocalBunchCrossing; //< bunchcrossing to be used in header
};

namespace
{
const BitSet sync(sampaSync().uint64(), 50);
}

template <typename CHARGESUM>
ElinkEncoder<BareFormat, CHARGESUM>::ElinkEncoder(uint8_t elinkId,
                                                  int phase)
  : mElinkId(elinkId),
    mBitSet{},
    mNofSync{0},
    mSyncIndex{0},
    mNofBitSeen{0},
    mLocalBunchCrossing{0},
    mPhase{phase}
{
  impl::assertIsInRange("elinkId", elinkId, 0, 39);

  // the phase is used to "simulate" a possible different timing alignment between elinks.

  if (phase < 0) {
    mPhase = static_cast<int>(rand() % 20);
  }

  for (int i = 0; i < mPhase; i++) {
    // filling the phase with random bits
    append(static_cast<bool>(rand() % 2));
  }
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::addChannelData(uint8_t chId, const std::vector<SampaCluster>& data)
{
  if (data.empty()) {
    throw std::invalid_argument("cannot add empty data");
  }
  assertSync();
  assertNotMixingClusters<CHARGESUM>(data);

  auto header = impl::buildSampaHeader(mElinkId, chId, data);

  append50(header.uint64());

  for (const auto& s : data) {
    append(s);
  }
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::addHeartbeat(uint20_t bunchCrossing)
{
  uint8_t chipAddress = impl::computeChipAddress(mElinkId, 0);
  SampaHeader sh = sampaHeartbeat(chipAddress, bunchCrossing);
  append50(sh.uint64());
}

/// append the data of a SampaCluster
template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::append(const SampaCluster& sc)
{
  append10(sc.nofSamples());
  append10(sc.sampaTime);
  appendCharges(sc);
}

/// append one bit (either set or unset)
template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::append(bool value)
{
  mBitSet.append(value);
  mNofBitSeen++;
}

/// append 10 bits (if value is more than 10 bits an exception is thrown)
template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::append10(uint16_t value)
{
  mBitSet.append(value, 10);
  mNofBitSeen += 10;
}

/// append 20 bits (if value is more than 20 bits an exception is thrown)
template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::append20(uint32_t value)
{
  mBitSet.append(value, 20);
  mNofBitSeen += 20;
}

/// append 50 bits (if value is more than 50 bits an exception is thrown)
template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::append50(uint64_t value)
{
  mBitSet.append(value, 50);
  mNofBitSeen += 50;
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::assertSync()
{
  bool firstSync = (mNofSync == 0);

  // if mSyncIndex is not zero it means
  // we have a pending sync to finish to transmit
  // (said otherwise we're not aligned to an expected 50bits mark)
  bool pendingSync = (mSyncIndex != 0);

  if (firstSync || pendingSync) {
    for (int i = mSyncIndex; i < 50; i++) {
      append(sync.get(i));
    }
    mSyncIndex = 0;
    if (firstSync) {
      mNofSync++;
    }
  }
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::clear()
{
  // we are not resetting the global counters mNofSync, mNofBitSeen,
  // just the bit stream
  mBitSet.clear();
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::fillWithSync(int upto)
{
  auto d = upto - len();
  mSyncIndex = circularAppend(mBitSet, sync, mSyncIndex, d);
  mNofSync += d / 50;
  mNofBitSeen += d;
}

template <typename CHARGESUM>
bool ElinkEncoder<BareFormat, CHARGESUM>::get(int i) const
{
  impl::assertIsInRange("i", i, 0, len() - 1);
  return mBitSet.get(i);
}

template <typename CHARGESUM>
int ElinkEncoder<BareFormat, CHARGESUM>::len() const
{
  return mBitSet.len();
}

template <typename CHARGESUM>
uint64_t ElinkEncoder<BareFormat, CHARGESUM>::range(int a, int b) const
{
  return mBitSet.subset(a, b).uint64(0, b - a + 1);
}

template <typename CHARGESUM>
void ElinkEncoder<BareFormat, CHARGESUM>::resetLocalBunchCrossing()
{
  mLocalBunchCrossing = mPhase;
}

} // namespace o2::mch::raw

#endif
