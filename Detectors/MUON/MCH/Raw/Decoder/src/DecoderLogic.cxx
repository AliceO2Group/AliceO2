// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DecoderLogic.h"
#include <sstream>
#include "Debug.h"
#include "MCHRawCommon/DataFormats.h"

namespace o2::mch::raw
{

DecoderLogic::DecoderLogic(DsElecId dsId, SampaChannelHandler sampaChannelHandler)
  : mDsId{dsId}, mSampaChannelHandler{sampaChannelHandler}
{
}

DsElecId DecoderLogic::dsId() const
{
  return mDsId;
}

std::ostream& DecoderLogic::debugHeader() const
{
  std::cout << fmt::format("--ULDEBUG--{:s}-----------", asString(mDsId));
  return std::cout;
}

bool DecoderLogic::hasError() const
{
  return mErrorMessage.has_value();
}

std::string DecoderLogic::errorMessage() const
{
  return hasError() ? mErrorMessage.value() : "";
}

void DecoderLogic::decrementClusterSize()
{
  --mClusterSize;
}

void DecoderLogic::setClusterSize(uint16_t value)
{
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
  debugHeader() << " -> size=" << mClusterSize << " maskIndex=" << mMaskIndex
                << " nof10BitWords=" << mSampaHeader.nof10BitWords()
                << " " << (hasError() ? "ERROR:" : "") << errorMessage() << "\n";
#endif
} // namespace o2::mch::raw

void DecoderLogic::setClusterTime(uint16_t value)
{
  mClusterTime = value;
#ifdef ULDEBUG
  debugHeader() << " -> time=" << mClusterTime << " maskIndex=" << mMaskIndex << "\n";
#endif
}

void DecoderLogic::reset()
{
#ifdef ULDEBUG
  debugHeader() << " -> reset\n";
#endif
  mMaskIndex = mMasks.size();
  mHeaderParts.clear();
  mClusterSize = 0;
  mNof10BitWords = 0;
  mErrorMessage = std::nullopt;
}

void DecoderLogic::setData(uint64_t data)
{
  mData = data;
  mMaskIndex = 0;
#ifdef ULDEBUG
  debugHeader() << fmt::format(">>>>> setData {:08X} maskIndex {} 10bits=", mData, mMaskIndex);
  for (int i = 0; i < mMasks.size(); i++) {
    std::cout << fmt::format("{:2d} ", data10(mData, i));
  }
  std::cout << "\n";
#endif
}

uint16_t DecoderLogic::data10(uint64_t value, size_t index) const
{
  uint64_t m = mMasks.at(index);
  return static_cast<uint16_t>((value & m) >> (index * 10) & 0x3FF);
}

uint16_t DecoderLogic::pop10()
{
  auto rv = data10(mData, mMaskIndex);
  mNof10BitWords = std::max(0, mNof10BitWords - 1);
  mMaskIndex = std::min(mMasks.size(), mMaskIndex + 1);
  return rv;
}

void DecoderLogic::sendCluster(const SampaCluster& sc) const
{
#ifdef ULDEBUG
  std::stringstream s;
  s << sc;
  debugHeader() << fmt::format(" calling channelHandler for {} ch {} = {}\n",
                               asString(mDsId), channelNumber64(mSampaHeader), s.str());
  debugHeader() << (*this) << "\n";
#endif
  mSampaChannelHandler(mDsId, channelNumber64(mSampaHeader), sc);
}

template <>
void DecoderLogic::addSample<SampleMode>(uint16_t sample)
{
#ifdef ULDEBUG
  debugHeader() << "sample = " << sample << "\n";
#endif
  mSamples.emplace_back(sample);

  if (mClusterSize == 0) {
    SampaCluster sc(mClusterTime, mSamples);
    sendCluster(sc);
    mSamples.clear();
  }
}

template <>
void DecoderLogic::addSample<ChargeSumMode>(uint16_t sample)
{
#ifdef ULDEBUG
  debugHeader() << "charge sum part = " << sample << "\n";
#endif
  mSamples.emplace_back(sample);

  if (mClusterSize == 0) {
    if (mSamples.size() != 2) {
      throw std::invalid_argument(fmt::format("expected sample size to be 2 but it is {}", mSamples.size()));
    }
    uint32_t q = (((static_cast<uint32_t>(mSamples[1]) & 0x3FF) << 10) | (static_cast<uint32_t>(mSamples[0]) & 0x3FF));
    SampaCluster sc(mClusterTime, q);
    sendCluster(sc);
    mSamples.clear();
  }
}

void DecoderLogic::completeHeader()
{
  uint64_t header{0};
  for (auto i = 0; i < mHeaderParts.size(); i++) {
    header += (static_cast<uint64_t>(mHeaderParts[i]) << (10 * i));
  }

  mSampaHeader = SampaHeader(header);
  mNof10BitWords = mSampaHeader.nof10BitWords();

#ifdef ULDEBUG
  debugHeader()
    << fmt::format(">>>>> completeHeader {:013X}\n", header)
    << "\n"
    << mSampaHeader << "\n";
#endif

  mHeaderParts.clear();
}

void DecoderLogic::addHeaderPart(uint16_t a)
{
  mHeaderParts.emplace_back(a);
#ifdef ULDEBUG
  debugHeader()
    << fmt::format(">>>>> readHeader {:08X}", a);
  for (auto h : mHeaderParts) {
    std::cout << fmt::format("{:4d} ", h);
  }
  std::cout << "\n";
#endif
}

bool DecoderLogic::moreDataAvailable() const
{
  return mMaskIndex < mMasks.size();
}

bool DecoderLogic::moreSampleToRead() const
{
  return mClusterSize > 0;
}

bool DecoderLogic::headerIsComplete() const
{
  return mHeaderParts.size() == 5;
}

bool DecoderLogic::moreWordsToRead() const
{
  return mNof10BitWords > 0;
}

std::ostream& operator<<(std::ostream& os, const DecoderLogic& ds)
{
  os << &ds << fmt::format(" DecoderLogic {} data=0X{:08X} n10={:4d} csize={:4d} ctime={:4d} maskIndex={:4d} ", asString(ds.dsId()), ds.mData, ds.mNof10BitWords, ds.mClusterSize, ds.mClusterTime, ds.mMaskIndex);
  os << fmt::format("header({})= ", ds.mHeaderParts.size());
  for (auto h : ds.mHeaderParts) {
    os << fmt::format("{:4d} ", h);
  }
  os << fmt::format("samples({})= ", ds.mSamples.size());
  for (auto s : ds.mSamples) {
    os << fmt::format("{:4d} ", s);
  }
  os << " handler=" << (ds.mSampaChannelHandler ? "valid" : "empty") << " ";

  os << fmt::format("moreWordsToRead: {} moreSampleToRead: {} moreDataAvailable: {}",
                    ds.moreWordsToRead(), ds.moreSampleToRead(), ds.moreDataAvailable());

  if (ds.hasError()) {
    os << " ERROR:" << ds.errorMessage();
  }
  return os;
}

} // namespace o2::mch::raw
