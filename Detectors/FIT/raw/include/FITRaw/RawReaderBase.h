// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawReaderBase.h base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for process

#ifndef ALICEO2_FIT_RAWREADERBASE_H_
#define ALICEO2_FIT_RAWREADERBASE_H_
#include <iostream>
#include <type_traits>
#include <vector>
#include <map>
#include <tuple>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/count.hpp>

#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include <Framework/Logger.h>

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
  typedef boost::mpl::vector<DataBlockTypes...> VecDataBlocks_t;
  std::tuple<std::vector<DataBlockTypes>...> mTupleVecDataBlocks;
  std::map<InteractionRecord, DigitBlock_t> mMapDigits;
  template <typename T>
  constexpr std::vector<T>& getVecDataBlocks()
  {
    typedef typename boost::mpl::find<VecDataBlocks_t, T>::type it_t;
    return std::get<it_t::pos::value>(mTupleVecDataBlocks);
  }
  //decoding binary data into data blocks
  template <class DataBlockType>
  size_t decodeBlocks(const gsl::span<const uint8_t> binaryPayload, std::vector<DataBlockType>& vecDataBlocks)
  {
    size_t srcPos = 0;
    while (srcPos < binaryPayload.size()) {
      auto& refDataBlock = vecDataBlocks.emplace_back();
      refDataBlock.decodeBlock(binaryPayload, srcPos);
      srcPos += refDataBlock.mSize;
      if (!refDataBlock.isCorrect()) {
        LOG(WARNING) << "INCORRECT DATA BLOCK! Byte position: " << srcPos - refDataBlock.mSize << " | Payload size: " << binaryPayload.size() << " | DataBlock size: " << refDataBlock.mSize;
        refDataBlock.print();
        vecDataBlocks.pop_back();
        return srcPos;
      }
    }
    return srcPos;
  }

  //processing data blocks into digits
  template <class DataBlockType, typename... T>
  void processBinaryData(gsl::span<const uint8_t> payload, T&&... feeParameters)
  {
    auto& vecDataBlocks = getVecDataBlocks<DataBlockType>();
    auto srcPos = decodeBlocks(payload, vecDataBlocks);
    for (auto& dataBlock : vecDataBlocks) {
      auto intRec = dataBlock.getInteractionRecord();
      auto [digitIter, isNew] = mMapDigits.try_emplace(intRec, intRec);
      digitIter->second.template processDigits<DataBlockType>(dataBlock, std::forward<T>(feeParameters)...);
    }
    vecDataBlocks.clear();
  }
  //pop digits
  template <class... DigitType>
  int getDigits(std::vector<DigitType>&... vecDigit)
  {
    int digitCounter = mMapDigits.size();
    for (auto& digit : mMapDigits) {
      digit.second.getDigits(vecDigit...);
    }
    mMapDigits.clear();
    return digitCounter;
  }

 private:
  //Check for unique DataBlock classes
  //Line below will not be compiled in case of duplicates among DataBlockTypes
  typedef std::void_t<std::enable_if_t<boost::mpl::count<boost::mpl::set<DataBlockTypes...>, DataBlockTypes>::value == 1>...> CheckUniqueTypes;
};

} // namespace fit
} // namespace o2

#endif
