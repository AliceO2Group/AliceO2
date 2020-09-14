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
//file DataBlockReader.h class  for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for proccess
//TODO:
//  in place contruction, change push_back to emplace_back in DataBlockReader
//  move semantic for DataBlocks

#ifndef ALICEO2_FIT_DATABLOCKREADER_H_
#define ALICEO2_FIT_DATABLOCKREADER_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FT0Raw/DataBlockRaw.h"
#include "FT0Raw/DigitBlockFT0.h"
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

#include <gsl/span>
namespace o2
{
namespace ft0
{

template <class DataBlockType>
class DataBlockReader
{
 public:
  DataBlockReader() = default;
  ~DataBlockReader() = default;
  typedef DataBlockType DataBlock;
  size_t decodeBlocks(const gsl::span<const uint8_t> binaryPayload, std::vector<DataBlockType>& vecDataBlocks)
  {
    size_t srcPos = 0;
    while (srcPos < binaryPayload.size()) { //checking element
      DataBlockType dataBlock;
      dataBlock.decodeBlock(binaryPayload, srcPos);
      srcPos += dataBlock.mSize;
      if (dataBlock.isCorrect())
        vecDataBlocks.push_back(dataBlock); //change to in-place construction? TODO
    }
    return srcPos;
  }
};

//Composition over multiple inheritance(multiple data block reader)
template <class DigitBlockType, class... DataBlockTypes>
class RawReaderBase : public boost::mpl::inherit<DataBlockReader<DataBlockTypes>...>::type
{
 public:
  typedef boost::mpl::vector<DataBlockTypes...> DataBlockVectorTypes;
  typedef boost::mpl::vector<DataBlockReader<DataBlockTypes>...> ReaderTypes;
  RawReaderBase() = default;
  ~RawReaderBase() = default;

  std::map<InteractionRecord, DigitBlockType> mMapDigits;
  template <class DataBlockType>
  void pushDataBlock(gsl::span<const uint8_t> payload, int linkID)
  {
    std::vector<DataBlockType> vecDataBlocks;
    auto srcPos = DataBlockReader<DataBlockType>::decodeBlocks(payload, vecDataBlocks);
    for (auto& dataBlock : vecDataBlocks) {
      auto intRec = dataBlock.getInteractionRecord();
      auto [digitIter, isNew] = mMapDigits.try_emplace(intRec, intRec);
      digitIter->second.template proccess<DataBlockType>(dataBlock, linkID);
    }
  }
};

// Raw reader for FT0
template <bool IsExtendedMode = false>
class RawReaderFT0 : public RawReaderBase<DigitBlockFT0, DataBlockPM, typename std::conditional<IsExtendedMode, DataBlockTCMext, DataBlockTCM>::type>
{
 public:
  typedef typename std::conditional<IsExtendedMode, DataBlockTCMext, DataBlockTCM>::type DataBlockTCMtype;
  typedef RawReaderBase<DigitBlockFT0, DataBlockPM, DataBlockTCMtype> RawReaderBaseType;

  RawReaderFT0() = default;
  ~RawReaderFT0() = default;
  //deserialize payload to raw data blocks and proccesss them to digits
  void proccess(int linkID, gsl::span<const uint8_t> payload)
  {
    if (0 <= linkID && linkID < 18) {
      //PM data proccessing
      RawReaderBaseType::template pushDataBlock<DataBlockPM>(payload, linkID);
    } else if (linkID == 18) {
      //TCM data proccessing
      RawReaderBaseType::template pushDataBlock<DataBlockTCMtype>(payload, linkID);
    } else {
      //put here code in case of bad rdh.linkID value
      LOG(INFO) << "WARNING! WRONG LINK ID!";
      return;
    }

    //
  }
  //pop digits
  int popDigits(std::vector<Digit>& vecDigit, std::vector<ChannelData>& vecChannelData)
  {
    int digitCounter = RawReaderBaseType::mMapDigits.size();
    for (auto& digit : (RawReaderBaseType::mMapDigits)) {
      digit.second.popData(vecDigit, vecChannelData);
    }
    (RawReaderBaseType::mMapDigits).clear();
    return digitCounter;
  }
};

} // namespace ft0
} // namespace o2

#endif