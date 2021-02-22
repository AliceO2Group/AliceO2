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
#include <vector>
#include <map>

#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include <Framework/Logger.h>

#include <gsl/span>
namespace o2
{
namespace fit
{
template <class DigitBlockType>
class RawReaderBase
{
 public:
  RawReaderBase() = default;
  ~RawReaderBase() = default;

  std::map<InteractionRecord, DigitBlockType> mMapDigits;

  //decoding binary data into data blocks
  template <class DataBlockType>
  size_t decodeBlocks(const gsl::span<const uint8_t> binaryPayload, std::vector<DataBlockType>& vecDataBlocks)
  {
    size_t srcPos = 0;
    while (srcPos < binaryPayload.size()) { //checking element
      DataBlockType dataBlock{};
      dataBlock.decodeBlock(binaryPayload, srcPos);
      srcPos += dataBlock.mSize;
      if (dataBlock.isCorrect()) {
        vecDataBlocks.push_back(dataBlock); //change to in-place construction? TODO
      } else {
        LOG(WARNING) << "INCORRECT DATA BLOCK! Byte position: " << srcPos - dataBlock.mSize << " | " << binaryPayload.size() << " | " << dataBlock.mSize;
        dataBlock.print();
      }
    }
    return srcPos;
  }

  //processing data blocks into digits
  template <class DataBlockType>
  void processBinaryData(gsl::span<const uint8_t> payload, int linkID)
  {
    std::vector<DataBlockType> vecDataBlocks;
    auto srcPos = decodeBlocks(payload, vecDataBlocks);

    for (auto& dataBlock : vecDataBlocks) {
      auto intRec = dataBlock.getInteractionRecord();
      auto [digitIter, isNew] = mMapDigits.try_emplace(intRec, intRec);
      digitIter->second.template process<DataBlockType>(dataBlock, linkID);
    }
  }
  /*
  void process(int linkID, gsl::span<const uint8_t> payload)
  {
    static_cast<RawReader*>(this)->processDigits(linkID,payload);
  }
  */
  //pop digits
  template <class... DigitType>
  int getDigits(std::vector<DigitType>&... vecDigit)
  {
    int digitCounter = mMapDigits.size();
    for (auto& digit : mMapDigits) {
      digit.second.pop(vecDigit...);
    }
    mMapDigits.clear();
    return digitCounter;
  }
};

} // namespace fit
} // namespace o2

#endif
