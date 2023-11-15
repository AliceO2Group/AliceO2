// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// file RawReaderBase.h base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
// Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for process

#ifndef ALICEO2_FIT_RAWREADERBASE_H_
#define ALICEO2_FIT_RAWREADERBASE_H_
#include <iostream>
#include <type_traits>
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/count.hpp>

#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include <Framework/Logger.h>
#include <FITRaw/DataBlockBase.h>
#include <DataFormatsFIT/RawDataMetric.h>

#include <gsl/span>
namespace o2
{
namespace fit
{
template <typename DigitBlockType, typename... DataBlockTypes>
class RawReaderBase
{
 public:
  RawReaderBase() = default;
  ~RawReaderBase() = default;
  typedef DigitBlockType DigitBlock_t;
  typedef RawDataMetric RawDataMetric_t;
  using LookupTable_t = typename DigitBlock_t::LookupTable_t;
  using EntryCRU_t = typename LookupTable_t::EntryCRU_t;
  using HashTableCRU_t = std::unordered_map<EntryCRU_t, RawDataMetric, typename LookupTable_t::MapEntryCRU2ModuleType_t::hasher, typename LookupTable_t::MapEntryCRU2ModuleType_t::key_equal>;

  HashTableCRU_t mHashTableMetrics{};
  typedef boost::mpl::vector<typename DataBlockTypes::DataBlockInvertedPadding_t..., DataBlockTypes...> VecDataBlocks_t;
  std::tuple<std::vector<typename DataBlockTypes::DataBlockInvertedPadding_t>..., std::vector<DataBlockTypes>...> mTupleVecDataBlocks;

  std::map<InteractionRecord, DigitBlock_t> mMapDigits;
  template <typename T>
  constexpr std::vector<T>& getVecDataBlocks()
  {
    typedef typename boost::mpl::find<VecDataBlocks_t, T>::type it_t;
    return std::get<it_t::pos::value>(mTupleVecDataBlocks);
  }
  // decoding binary data into data blocks
  template <class DataBlockType>
  size_t decodeBlocks(const gsl::span<const uint8_t> binaryPayload, RawDataMetric& metric, std::vector<DataBlockType>& vecDataBlocks)
  {
    size_t srcPos = 0;
    const auto& payloadSize = binaryPayload.size();
    const int nWords = payloadSize / DataBlockType::DataBlockWrapperHeader_t::sSizeWord;
    const int pageSizeThreshold = DataBlockType::DataBlockWrapperHeader_t::sSizeWord * (nWords - int(nWords > 1)); // no need in reading last GBT word, this will be 0xff... or empty header
    while (srcPos < pageSizeThreshold) {
      auto& refDataBlock = vecDataBlocks.emplace_back();
      refDataBlock.decodeBlock(binaryPayload, srcPos);
      srcPos += refDataBlock.mSize;
      if (metric.checkBadDataBlock(refDataBlock.mStatusBitsAll)) {
        // exclude data block in case of single header(no data, total size == 16 bytes)
        vecDataBlocks.pop_back();
      }
    }
    return srcPos;
  }

  // processing data blocks into digits
  template <class DataBlockType, typename... T>
  void processBinaryData(gsl::span<const uint8_t> payload, uint16_t feeID, uint8_t linkID, uint8_t epID)
  {
    auto& vecDataBlocks = getVecDataBlocks<DataBlockType>();
    auto& metric = addMetric(feeID, linkID, epID);
    auto srcPos = decodeBlocks(payload, metric, vecDataBlocks);
    for (const auto& dataBlock : vecDataBlocks) {
      auto intRec = dataBlock.getInteractionRecord();
      auto [digitIter, isNew] = mMapDigits.try_emplace(intRec, intRec);
      digitIter->second.template processDigits<DataBlockType>(dataBlock, metric, static_cast<int>(linkID), static_cast<int>(epID));
      metric.addStatusBit(RawDataMetric::EStatusBits::kDecodedDataBlock);
    }
    vecDataBlocks.clear();
  }
  RawDataMetric& addMetric(uint16_t feeID, uint8_t linkID, uint8_t epID, bool isRegisteredFEE = true)
  {
    auto metricPair = mHashTableMetrics.try_emplace(EntryCRU_t{static_cast<int>(linkID), static_cast<int>(epID)}, linkID, epID, feeID, isRegisteredFEE);
    auto& metric = metricPair.first->second;
    return metric;
  }
  // pop digits
  template <typename... VecDigitType>
  int getDigits(VecDigitType&... vecDigit)
  {
    int digitCounter = mMapDigits.size();
    for (auto& digit : mMapDigits) {
      digit.second.getDigits(vecDigit...);
    }
    mMapDigits.clear();
    return digitCounter;
  }
  void getMetrics(std::vector<RawDataMetric>& vecMetrics)
  {
    for (const auto& en : mHashTableMetrics) {
      vecMetrics.push_back(en.second);
    }
    mHashTableMetrics.clear();
  }

 private:
  // Check for unique DataBlock classes
  // Line below will not be compiled in case of duplicates among DataBlockTypes
  typedef std::void_t<std::enable_if_t<boost::mpl::count<boost::mpl::set<DataBlockTypes...>, DataBlockTypes>::value == 1>...> CheckUniqueTypes;
};

} // namespace fit
} // namespace o2

#endif
