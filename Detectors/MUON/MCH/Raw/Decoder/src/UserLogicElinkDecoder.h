// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_USER_LOGIC_ELINK_DECODER_H
#define O2_MCH_RAW_USER_LOGIC_ELINK_DECODER_H

#include "Debug.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawDecoder/DecodedDataHandlers.h"
#include "MCHRawDecoder/ErrorCodes.h"
#include "MCHRawElecMap/DsElecId.h"
#include <bitset>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <set>

namespace o2::mch::raw
{

template <typename CHARGESUM>
class UserLogicElinkDecoder
{
 public:
  UserLogicElinkDecoder(DsElecId dsId, DecodedDataHandlers decodedDataHandlers);

  /// Append 50 bits-worth of data
  void append(uint64_t data50, uint8_t error, bool incomplete);

  /// Reset our internal state
  /// i.e. assume the sync has to be found again
  void reset();

 private:
  /// The possible states we can be in
  enum class State : int {
    WaitingSync,
    WaitingHeader,
    WaitingSize,
    WaitingTime,
    WaitingSample
  };
  std::string asString(State state) const;

  using uint10_t = uint16_t;

  template <typename T>
  friend std::ostream& operator<<(std::ostream& os, const o2::mch::raw::UserLogicElinkDecoder<T>& e);

  const DsElecId& dsId() const { return mDsId; }

  void clear();
  bool hasError() const;
  bool isHeaderComplete() const { return mHeaderParts.size() == 5; }
  bool moreSampleToRead() const { return mSamplesToRead > 0; }
  bool moreWordsToRead() const { return mNof10BitWords > 0; }
  std::ostream& debugHeader() const;
  std::string errorMessage() const;
  bool append10(uint10_t data10);
  void completeHeader();
  void oneLess10BitWord();
  void prepareAndSendCluster();
  void sendCluster(const SampaCluster& sc) const;
  void sendHBPacket();
  void sendError(int8_t chip, uint32_t error) const;
  void setClusterSize(uint10_t value);
  void setClusterTime(uint10_t value);
  void setHeaderPart(uint10_t data10);
  void setSample(uint10_t value);
  void transition(State to);

 private:
  DsElecId mDsId;
  DecodedDataHandlers mDecodedDataHandlers;
  State mState;
  std::vector<uint10_t> mSamples{};
  std::vector<uint10_t> mHeaderParts{};
  SampaHeader mSampaHeader{};
  uint10_t mNof10BitWords{};
  uint10_t mClusterSize{};
  uint10_t mSamplesToRead{};
  uint10_t mClusterTime{};
  std::optional<std::string> mErrorMessage{std::nullopt};
};

constexpr bool isSync(uint64_t data)
{
  constexpr uint64_t sampaSyncWord{0x1555540f00113};
  return data == sampaSyncWord;
};

template <typename CHARGESUM>
UserLogicElinkDecoder<CHARGESUM>::UserLogicElinkDecoder(DsElecId dsId,
                                                        DecodedDataHandlers decodedDataHandlers)
  : mDsId{dsId}, mDecodedDataHandlers{decodedDataHandlers}, mState{State::WaitingSync}
{
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::append(uint64_t data50, uint8_t error, bool incomplete)
{
#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> append50 {:013x} error {} incomplete {} data10={:d} {:d} {:d} {:d} {:d}\n", data50, error, incomplete, static_cast<uint10_t>(data50 & 0x3FF), static_cast<uint10_t>((data50 & 0xFFC00) >> 10), static_cast<uint10_t>((data50 & 0x3FF00000) >> 20), static_cast<uint10_t>((data50 & 0xFFC0000000) >> 30), static_cast<uint10_t>((data50 & 0x3FF0000000000) >> 40));
#endif

  if (isSync(data50)) {
#ifdef ULDEBUG
    debugHeader() << (*this) << fmt::format(" --> SYNC word found {:013x}\n", data50);
#endif
    clear();
    transition(State::WaitingHeader);
    return;
  }

  auto data = data50;

  int i;
  for (i = 0; i < 5; i++) {
    bool packetEnd = append10(static_cast<uint10_t>(data & 0x3FF));
    data >>= 10;
#ifdef ULDEBUG
    if (incomplete) {
      debugHeader() << (*this) << fmt::format(" --> incomplete {} packetEnd @i={}\n", incomplete, packetEnd, i);
    }
#endif
    if (hasError()) {
#ifdef ULDEBUG
      debugHeader() << (*this) << " reset due to hasError\n";
#endif
      reset();
      break;
    }
    if (incomplete && packetEnd) {
#ifdef ULDEBUG
      debugHeader() << (*this) << " stop due to isIncomplete\n";
#endif
      break;
    }
  }

  if (incomplete && (i == 5) && (mState != State::WaitingSync)) {
#ifdef ULDEBUG
    debugHeader() << (*this) << " data packet end not found when isIncomplete --> resetting\n";
#endif
    sendError(static_cast<int8_t>(mSampaHeader.chipAddress()), static_cast<uint32_t>(ErrorBadIncompleteWord));
    reset();
  }
} // namespace o2::mch::raw

template <typename CHARGESUM>
struct DataFormatSizeFactor;

template <>
struct DataFormatSizeFactor<SampleMode> {
  static constexpr uint8_t value = 1;
};

template <>
struct DataFormatSizeFactor<ChargeSumMode> {
  static constexpr uint8_t value = 2;
};

template <typename CHARGESUM>
bool UserLogicElinkDecoder<CHARGESUM>::append10(uint10_t data10)
{
  bool result = false;
#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> data10 {:d}\n", data10);
#endif
  switch (mState) {
    case State::WaitingHeader:
      setHeaderPart(data10);
      if (isHeaderComplete()) {
        completeHeader();
        if (isSync(mSampaHeader.uint64())) {
          reset();
        } else if (mSampaHeader.packetType() == SampaPacketType::HeartBeat) {
          if (mSampaHeader.isHeartbeat()) {
            sendHBPacket();
            transition(State::WaitingHeader);
            result = true;
          } else {
            mErrorMessage = "badly formatted Heartbeat packet";
            sendError(static_cast<int8_t>(mSampaHeader.chipAddress()), static_cast<uint32_t>(ErrorBadHeartBeatPacket));
            reset();
          }
        } else {
          if (mSampaHeader.nof10BitWords() > 2) {
            transition(State::WaitingSize);
          } else {
            reset();
          }
        }
      }
      break;
    case State::WaitingSize:
      if (moreWordsToRead()) {
        setClusterSize(data10);
        if (hasError()) {
          return false;
        }
        transition(State::WaitingTime);
      } else {
        mErrorMessage = "WaitingSize but no more words";
        return false;
      }
      break;
    case State::WaitingTime:
      if (moreWordsToRead()) {
        setClusterTime(data10);
        transition(State::WaitingSample);
      } else {
        mErrorMessage = "WaitingTime but no more words";
        return false;
      }
      break;
    case State::WaitingSample:
      if (moreSampleToRead()) {
        setSample(data10);
      }
      if (!moreSampleToRead()) {
        if (moreWordsToRead()) {
          transition(State::WaitingSize);
        } else {
          transition(State::WaitingHeader);
          result = true;
        }
      }
      break;
    default:
      break;
  };
  return result;
}

template <typename CHARGESUM>
std::string UserLogicElinkDecoder<CHARGESUM>::asString(State s) const
{
  switch (s) {
    case State::WaitingSync:
      return "WaitingSync";
      break;
    case State::WaitingHeader:
      return "WaitingHeader";
      break;
    case State::WaitingSize:
      return "WaitingSize";
      break;
    case State::WaitingTime:
      return "WaitingTime";
      break;
    case State::WaitingSample:
      return "WaitingSample";
      break;
  };
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::clear()
{
  mSamples.clear();
  mHeaderParts.clear();
  mNof10BitWords = 0;
  mClusterSize = 0;
  mErrorMessage = std::nullopt;
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::completeHeader()
{
  uint64_t header{0};
  for (auto i = 0; i < mHeaderParts.size(); i++) {
    header += (static_cast<uint64_t>(mHeaderParts[i]) << (10 * i));
  }

  mSampaHeader = SampaHeader(header);
  mNof10BitWords = mSampaHeader.nof10BitWords();

#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> completeHeader {:013X}\n", header);
  debugHeader() << "\n";
  std::stringstream s(o2::mch::raw::asString(mSampaHeader));
  std::string part;
  while (std::getline(s, part, '\n')) {
    debugHeader() << (*this) << part << "\n";
  }
  debugHeader() << "\n";
#endif

  mHeaderParts.clear();
}

template <typename CHARGESUM>
std::ostream& UserLogicElinkDecoder<CHARGESUM>::debugHeader() const
{
  return std::cout << "---";
}

template <typename CHARGESUM>
std::string UserLogicElinkDecoder<CHARGESUM>::errorMessage() const
{
  return hasError() ? mErrorMessage.value() : "";
}

template <typename CHARGESUM>
bool UserLogicElinkDecoder<CHARGESUM>::hasError() const
{
  return mErrorMessage.has_value();
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::reset()
{
#ifdef ULDEBUG
  debugHeader() << (*this) << " ---> reset\n";
#endif
  clear();
  transition(State::WaitingSync);
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::sendCluster(const SampaCluster& sc) const
{
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> "
                << fmt::format(" calling channelHandler for {} ch {} = {}\n",
                               o2::mch::raw::asString(mDsId),
                               getDualSampaChannelId(mSampaHeader),
                               o2::mch::raw::asString(sc));
#endif
  mDecodedDataHandlers.sampaChannelHandler(mDsId, getDualSampaChannelId(mSampaHeader), sc);
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::sendError(int8_t chip, uint32_t error) const
{
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> "
                << fmt::format(" calling errorHandler for {} chip {} = {}\n",
                               o2::mch::raw::asString(mDsId), chip, error);
#endif
  SampaErrorHandler handler = mDecodedDataHandlers.sampaErrorHandler;
  if (handler) {
    handler(mDsId, chip, error);
  }
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::oneLess10BitWord()
{
  mNof10BitWords = std::max(0, mNof10BitWords - 1);
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::setClusterSize(uint10_t value)
{
  oneLess10BitWord();
  mClusterSize = value;
  if (CHARGESUM()()) {
    mSamplesToRead = 2;
  } else {
    mSamplesToRead = mClusterSize;
  }
  int checkSize = mSamplesToRead + 2 - mSampaHeader.nof10BitWords();
  mErrorMessage = std::nullopt;
  if (mClusterSize == 0) {
    mErrorMessage = "cluster size is zero";
    sendError(static_cast<int8_t>(mSampaHeader.chipAddress()), static_cast<uint32_t>(ErrorBadClusterSize));
  }
  if (checkSize > 0) {
    mErrorMessage = "number of samples bigger than nof10BitWords";
    sendError(static_cast<int8_t>(mSampaHeader.chipAddress()), static_cast<uint32_t>(ErrorBadClusterSize));
  }
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> size=" << mClusterSize << "  samples=" << mSamplesToRead << "\n";
#endif
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::setClusterTime(uint10_t value)
{
  oneLess10BitWord();
  mClusterTime = value;
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> time=" << mClusterTime << "\n";
#endif
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::setHeaderPart(uint10_t a)
{
  oneLess10BitWord();
  mHeaderParts.emplace_back(a);
#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> readHeader {:08X}\n", a);
#endif
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::setSample(uint10_t sample)
{
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> sample = " << sample << "\n";
#endif
  --mSamplesToRead;
  oneLess10BitWord();
  mSamples.emplace_back(sample);

  if (mSamplesToRead == 0) {
    prepareAndSendCluster();
  }
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::sendHBPacket()
{
  SampaHeartBeatHandler handler = mDecodedDataHandlers.sampaHeartBeatHandler;
  if (handler) {
    handler(mDsId, mSampaHeader.chipAddress() % 2, mSampaHeader.bunchCrossingCounter());
  }
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::transition(State to)
{
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> Transition from " << asString(mState) << " to " << asString(to) << "\n";
#endif
  mState = to;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const o2::mch::raw::UserLogicElinkDecoder<T>& e)
{
  os << fmt::format("{} n10={:4d} size={:4d} t={:4d} ", asString(e.mDsId), e.mNof10BitWords, e.mClusterSize, e.mClusterTime);
  os << fmt::format("h({:2d})= ", e.mHeaderParts.size());
  for (auto h : e.mHeaderParts) {
    os << fmt::format("{:4d} ", h);
  }
  os << fmt::format("s({:2d})= ", e.mSamples.size());
  for (auto s : e.mSamples) {
    os << fmt::format("{:4d} ", s);
  }
  if (!e.mDecodedDataHandlers.sampaChannelHandler) {
    os << " empty handler ";
  }

  os << fmt::format("moreWords: {:5} moreSample: {:5} ",
                    e.moreWordsToRead(), e.moreSampleToRead());

  if (e.hasError()) {
    os << " ERROR:" << e.errorMessage();
  }
  return os;
}

} // namespace o2::mch::raw

#endif
