// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_BARE_ELINK_DECODER_H
#define O2_MCH_RAW_BARE_ELINK_DECODER_H

#include "Assertions.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawDecoder/DecodedDataHandlers.h"
#include <bitset>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace o2::mch::raw
{

/// @brief Main element of the MCH Bare Raw Data Format decoder.
///
/// A BareElinkDecoder manages the bit stream for one Elink.
///
/// Bits coming from parts of the GBT words are added to the Elink using the
/// append() method and each time a SampaCluster is decoded,
/// it is passed to the DecodedDataHandlers for further processing (or none).
///
/// \nosubgrouping
///
template <typename CHARGESUM>
class BareElinkDecoder
{
 public:
  /// Constructor.
  /// \param dsId the (electronic) id of the dual sampa this elink
  /// is connected  to
  /// \param decodedDataHandlers a structure with various callable that
  /// handle the Sampa packets and decoding errors
  BareElinkDecoder(DsElecId dsId, DecodedDataHandlers decodedDataHandlers);

  /** @name Main interface
  */
  ///@{

  /// Append two bits (from the same dual sampa, one per sampa) to the Elink.
  void append(bool bit0, bool bit1);
  ///@}

  // /// linkId is the GBT id this Elink is part of
  // uint8_t linkId() const;

  /** @name Methods for testing
    */
  ///@{

  /// Current number of bits we're holding
  int len() const;

  /// Reset our internal bit stream, and the sync status
  /// i.e. assume the sync has to be found again
  void reset();
  ///@}

 private:
  /// The possible states we can be in
  enum class State : int {
    LookingForSync,    //< we've not found a sync yet
    LookingForHeader,  //< we've looking for a 50-bits header
    ReadingNofSamples, //< we're (about to) read nof of samples
    ReadingTimestamp,  //< we're (about to) read a timestamp (for the current cluster)
    ReadingSample,     //< we're (about to) read a sample (for the current cluster)
    ReadingClusterSum  //< we're (about to) read a chargesum (for the current cluster)
  };

  std::string name(State state) const;
  void appendOneBit(bool bit);
  void changeState(State newState, int newCheckpoint);
  void changeToReadingData();
  void clear(int checkpoint);
  void findSync();
  void handlReadClusterSum();
  void handleHeader();
  void handleReadClusterSum();
  void handleReadData();
  void handleReadSample();
  void handleReadTimestamp();
  void oneLess10BitWord();
  void process();
  void sendCluster();
  void sendHBPacket();
  void softReset();

  template <typename T>
  friend std::ostream& operator<<(std::ostream& os, const o2::mch::raw::BareElinkDecoder<T>& e);

 private:
  DsElecId mDsId;
  DecodedDataHandlers mDecodedDataHandlers; //< The structure with the callables that  deal with the Sampa packets and the decoding errors
  SampaHeader mSampaHeader;                 //< Current SampaHeader
  uint64_t mBitBuffer;                      //< Our internal bit stream buffer
  /** @name internal global counters
    */

  ///@{
  uint64_t mNofSync;               //< Number of SYNC words we've seen so far
  uint64_t mNofBitSeen;            //< Total number of bits seen
  uint64_t mNofHeaderSeen;         //< Total number of headers seen
  uint64_t mNofHammingErrors;      //< Total number of hamming errors seen
  uint64_t mNofHeaderParityErrors; //< Total number of header parity errors seen
  ///@}

  uint64_t mCheckpoint;           //< mask of the next state transition check to be done in process()
  uint16_t mNof10BitsWordsToRead; //< number of 10 bits words to be read

  uint10_t mClusterSize;
  uint16_t mNofSamples;
  uint16_t mTimestamp;
  std::vector<uint16_t> mSamples;
  uint32_t mClusterSum;
  uint64_t mMask;

  State mState; //< the state we are in
};

constexpr int HEADERSIZE = 50;

std::string bitBufferString(const std::bitset<50>& bs, int imax);

template <typename CHARGESUM>
BareElinkDecoder<CHARGESUM>::BareElinkDecoder(DsElecId dsId,
                                              DecodedDataHandlers decodedDataHandlers)
  : mDsId{dsId},
    mDecodedDataHandlers{decodedDataHandlers},
    mSampaHeader{},
    mBitBuffer{},
    mNofSync{},
    mNofBitSeen{},
    mNofHeaderSeen{},
    mNofHammingErrors{},
    mNofHeaderParityErrors{},
    mCheckpoint{(static_cast<uint64_t>(1) << HEADERSIZE)},
    mNof10BitsWordsToRead{},
    mClusterSize{},
    mNofSamples{},
    mTimestamp{},
    mSamples{},
    mClusterSum{},
    mState{State::LookingForSync},
    mMask{1}
{
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::appendOneBit(bool bit)
{
  mNofBitSeen++;

  mBitBuffer += bit * mMask;
  mMask *= 2;

  if (mMask == mCheckpoint) {
    process();
  }
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::append(bool bit0, bool bit1)
{
  appendOneBit(bit0);
  appendOneBit(bit1);
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::changeState(State newState, int newCheckpoint)
{
  mState = newState;
  clear(newCheckpoint);
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::clear(int checkpoint)
{
  mBitBuffer = 0;
  mCheckpoint = static_cast<uint64_t>(1) << checkpoint;
  mMask = 1;
}

/// findSync checks if the last 50 bits of the bit stream
/// match the Sampa SYNC word.
///
/// - if they are then reset the bit stream and sets the checkpoint to 50 bits
/// - if they are not then pop the first bit out
template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::findSync()
{
  assert(mState == State::LookingForSync);
  if (mBitBuffer != sampaSyncWord) {
    mBitBuffer >>= 1;
    mMask /= 2;
    return;
  }
  changeState(State::LookingForHeader, HEADERSIZE);
  mNofSync++;
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::handleHeader()
{
  assert(mState == State::LookingForHeader);

  mSampaHeader.uint64(mBitBuffer);

  ++mNofHeaderSeen;

  if (mSampaHeader.hasError()) {
    ++mNofHammingErrors;
  }

  switch (mSampaHeader.packetType()) {
    case SampaPacketType::DataTruncated:
    case SampaPacketType::DataTruncatedTriggerTooEarly:
    case SampaPacketType::DataTriggerTooEarly:
    case SampaPacketType::DataTriggerTooEarlyNumWords:
    case SampaPacketType::DataNumWords:
      // data with a problem is still data, i.e. there will
      // probably be some data words to read in...
      // so we fallthrough the simple Data case
    case SampaPacketType::Data:
      mNof10BitsWordsToRead = mSampaHeader.nof10BitWords();
      changeState(State::ReadingNofSamples, 10);
      break;
    case SampaPacketType::Sync:
      mNofSync++;
      softReset();
      break;
    case SampaPacketType::HeartBeat:
      if (mSampaHeader.isHeartbeat()) {
        sendHBPacket();
        changeState(State::LookingForHeader, HEADERSIZE);
      } else {
        softReset();
      }
      break;
    default:
      throw std::logic_error("that should not be possible");
      break;
  }
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::handleReadClusterSum()
{
  mClusterSum = mBitBuffer;
  oneLess10BitWord();
  oneLess10BitWord();
  sendCluster();
  if (mNof10BitsWordsToRead) {
    changeState(State::ReadingNofSamples, 10);
  } else {
    changeState(State::LookingForHeader, HEADERSIZE);
  }
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::handleReadData()
{
  assert(mState == State::ReadingTimestamp || mState == State::ReadingSample);
  if (mState == State::ReadingTimestamp) {
    mTimestamp = mBitBuffer;
  }
  oneLess10BitWord();
  changeToReadingData();
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::handleReadSample()
{
  mSamples.push_back(mBitBuffer);
  if (mNofSamples > 0) {
    --mNofSamples;
  }
  oneLess10BitWord();
  if (mNofSamples) {
    changeToReadingData();
  } else {
    sendCluster();
    if (mNof10BitsWordsToRead) {
      changeState(State::ReadingNofSamples, 10);
    } else {
      changeState(State::LookingForHeader, HEADERSIZE);
    }
  }
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::handleReadTimestamp()
{
  assert(mState == State::ReadingNofSamples);
  oneLess10BitWord();
  mNofSamples = mBitBuffer;
  mClusterSize = mNofSamples;
  changeState(State::ReadingTimestamp, 10);
}

template <typename CHARGESUM>
int BareElinkDecoder<CHARGESUM>::len() const
{
  return static_cast<int>(std::floor(log2(1.0 * mMask)) + 1);
}

template <typename CHARGESUM>
std::string BareElinkDecoder<CHARGESUM>::name(State s) const
{
  switch (s) {
    case State::LookingForSync:
      return "LookingForSync";
      break;
    case State::LookingForHeader:
      return "LookingForHeader";
      break;
    case State::ReadingNofSamples:
      return "ReadingNofSamples";
      break;
    case State::ReadingTimestamp:
      return "ReadingTimestamp";
      break;
    case State::ReadingSample:
      return "ReadingSample";
      break;
    case State::ReadingClusterSum:
      return "ReadingClusterSum";
      break;
  };
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::oneLess10BitWord()
{
  if (mNof10BitsWordsToRead > 0) {
    --mNof10BitsWordsToRead;
  }
}

/// process the bit stream content.
template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::process()
{
  switch (mState) {
    case State::LookingForSync:
      findSync();
      break;
    case State::LookingForHeader:
      handleHeader();
      break;
    case State::ReadingNofSamples:
      handleReadTimestamp();
      break;
    case State::ReadingTimestamp:
      handleReadData();
      break;
    case State::ReadingSample:
      handleReadSample();
      break;
    case State::ReadingClusterSum:
      handleReadClusterSum();
      break;
  }
};

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::softReset()
{
  clear(HEADERSIZE);
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::reset()
{
  softReset();
  mState = State::LookingForSync;
}

template <typename CHARGESUM>
std::ostream& operator<<(std::ostream& os, const o2::mch::raw::BareElinkDecoder<CHARGESUM>& e)
{
  os << fmt::format("ID{:2d} cruId {:2d} sync {:6d} cp 0x{:6x} mask 0x{:6x} state {:17s} len {:6d} nseen {:6d} errH {:6} errP {:6} head {:6d} n10w {:6d} nsamples {:6d} mode {} bbuf {:s}",
                    e.mLinkId, e.mCruId, e.mNofSync, e.mCheckpoint, e.mMask,
                    e.name(e.mState),
                    e.len(), e.mNofBitSeen,
                    e.mNofHeaderSeen,
                    e.mNofHammingErrors,
                    e.mNofHeaderParityErrors,
                    e.mNof10BitsWordsToRead,
                    e.mNofSamples,
                    (e.mClusterSumMode ? "CLUSUM" : "SAMPLE"),
                    bitBufferString(e.mBitBuffer, e.mMask));
  return os;
}

template <typename CHARGESUM>
void BareElinkDecoder<CHARGESUM>::sendHBPacket()
{
  SampaHeartBeatHandler handler = mDecodedDataHandlers.sampaHeartBeatHandler;
  if (handler) {
    handler(mDsId, mSampaHeader.chipAddress() % 2, mSampaHeader.bunchCrossingCounter());
  }
}

} // namespace o2::mch::raw

#endif
