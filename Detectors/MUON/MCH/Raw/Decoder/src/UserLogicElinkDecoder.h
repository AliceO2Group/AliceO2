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
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawElecMap/DsElecId.h"
#include <bitset>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

namespace o2::mch::raw
{

template <typename CHARGESUM>
class UserLogicElinkDecoder
{
 public:
  UserLogicElinkDecoder(DsElecId dsId, SampaChannelHandler sampaChannelHandler);

  /// Append 50 bits-worth of data
  void append(uint64_t data50, uint8_t error);

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

  void clear();
  bool hasError() const;
  bool isHeaderComplete() const { return mHeaderParts.size() == 5; }
  bool moreSampleToRead() const { return mClusterSize > 0; }
  bool moreWordsToRead() const { return mNof10BitWords > 0; }
  std::ostream& debugHeader() const;
  std::string errorMessage() const;
  void append10(uint10_t data10);
  void completeHeader();
  void oneLess10BitWord();
  void prepareAndSendCluster();
  void sendCluster(const SampaCluster& sc) const;
  void setClusterSize(uint10_t value);
  void setClusterTime(uint10_t value);
  void setHeaderPart(uint10_t data10);
  void setSample(uint10_t value);
  void transition(State to);

 private:
  DsElecId mDsId;
  SampaChannelHandler mSampaChannelHandler;
  State mState;
  std::vector<uint10_t> mSamples{};
  std::vector<uint10_t> mHeaderParts{};
  SampaHeader mSampaHeader{};
  uint10_t mNof10BitWords{};
  uint10_t mClusterSize{};
  uint10_t mClusterTime{};
  std::optional<std::string> mErrorMessage{std::nullopt};
};

constexpr bool isSync(uint64_t data)
{
  constexpr uint64_t sampaSyncWord{0x1555540f00113};
  return data == sampaSyncWord;
};

constexpr bool isIncomplete(uint8_t error)
{
  return (error & 0x4) > 0;
}

template <typename CHARGESUM>
UserLogicElinkDecoder<CHARGESUM>::UserLogicElinkDecoder(DsElecId dsId,
                                                        SampaChannelHandler sampaChannelHandler)
  : mDsId{dsId}, mSampaChannelHandler{sampaChannelHandler}, mState{State::WaitingSync}
{
}

template <typename CHARGESUM>
void UserLogicElinkDecoder<CHARGESUM>::append(uint64_t data50, uint8_t error)
{
#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> append50 {:013x} error {} data10={:d} {:d} {:d} {:d}\n", data50, error, static_cast<uint10_t>(data50 & 0x3FF), static_cast<uint10_t>((data50 & 0xFFC00) >> 10), static_cast<uint10_t>((data50 & 0x3FF00000) >> 20), static_cast<uint10_t>((data50 & 0xFFC0000000) >> 30), static_cast<uint10_t>((data50 & 0x3FF0000000000) >> 40));
#endif

  if (isSync(data50)) {
    clear();
    transition(State::WaitingHeader);
    return;
  }

  auto data = data50;

  for (auto i = 0; i < 5; i++) {
    append10(static_cast<uint10_t>(data & 0x3FF));
    data >>= 10;
    if (hasError()) {
#ifdef ULDEBUG
      debugHeader() << (*this) << " reset due to hasError\n";
#endif
      reset();
      break;
    }
    if (mState == State::WaitingHeader && isIncomplete(error)) {
#ifdef ULDEBUG
      debugHeader() << (*this) << " reset due to isIncomplete\n";
#endif
      reset();
      return;
    }
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
void UserLogicElinkDecoder<CHARGESUM>::append10(uint10_t data10)
{
#ifdef ULDEBUG
  debugHeader() << (*this) << fmt::format(" --> data10 {:d}\n", data10);
#endif
  switch (mState) {
    case State::WaitingHeader:
      setHeaderPart(data10);
      if (isHeaderComplete()) {
        completeHeader();
        if (isSync(mSampaHeader.uint64()) ||
            mSampaHeader.packetType() == SampaPacketType::HeartBeat ||
            mSampaHeader.nof10BitWords() == 0) {
          reset();
        } else {
          transition(State::WaitingSize);
        }
      }
      break;
    case State::WaitingSize:
      if (moreWordsToRead()) {
        auto factor = DataFormatSizeFactor<CHARGESUM>::value;
        auto value = factor * data10;
        setClusterSize(value);
        if (hasError()) {
          return;
        }
        transition(State::WaitingTime);
      } else {
        mErrorMessage = "WaitingSize but no more words";
        return;
      }
      break;
    case State::WaitingTime:
      if (moreWordsToRead()) {
        setClusterTime(data10);
        transition(State::WaitingSample);
      } else {
        mErrorMessage = "WaitingTime but no more words";
        return;
      }
      break;
    case State::WaitingSample:
      if (moreSampleToRead()) {
        setSample(data10);
      } else if (moreWordsToRead()) {
        transition(State::WaitingSize);
        append10(data10);
      } else {
        transition(State::WaitingHeader);
        append10(data10);
      }
      break;
    default:
      break;
  };
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
  //std::cout << fmt::format("--ULDEBUG--{:p}-----------", reinterpret_cast<const void*>(this));
  //return std::cout;
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
                               channelNumber64(mSampaHeader),
                               o2::mch::raw::asString(sc));
#endif
  mSampaChannelHandler(mDsId, channelNumber64(mSampaHeader), sc);
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
  int checkSize = mClusterSize + 2 - mSampaHeader.nof10BitWords();
  mErrorMessage = std::nullopt;
  if (mClusterSize == 0) {
    mErrorMessage = "cluster size is zero";
  }
  if (checkSize > 0) {
    mErrorMessage = "cluster size bigger than nof10BitWords";
  }
#ifdef ULDEBUG
  debugHeader() << (*this) << " --> size=" << mClusterSize << "\n";
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
  --mClusterSize;
  oneLess10BitWord();
  mSamples.emplace_back(sample);

  if (mClusterSize == 0) {
    prepareAndSendCluster();
  }
}

template <>
void UserLogicElinkDecoder<SampleMode>::prepareAndSendCluster()
{
  if (mSampaChannelHandler) {
    SampaCluster sc(mClusterTime, mSamples);
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
  uint32_t q = (((static_cast<uint32_t>(mSamples[1]) & 0x3FF) << 10) | (static_cast<uint32_t>(mSamples[0]) & 0x3FF));
  SampaCluster sc(mClusterTime, q);
  sendCluster(sc);
  mSamples.clear();
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
  if (!e.mSampaChannelHandler) {
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
