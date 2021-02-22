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
//file DigitBlockFT0.h class  for proccessing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch
// TODO:
//  traites for DataBlocks
//  check if the EventID filling is correct

#ifndef ALICEO2_FIT_DIGITBLOCKFT0_H_
#define ALICEO2_FIT_DIGITBLOCKFT0_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <Rtypes.h>
#include "FT0Raw/DataBlockFT0.h"
#include "FITRaw/DigitBlockBase.h"

#include <CommonDataFormat/InteractionRecord.h>
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/LookUpTable.h"

#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>

#include <gsl/span>

using namespace o2::fit;

namespace o2
{
namespace ft0
{
//Normal data taking mode
class DigitBlockFT0 : public DigitBlockBase<DigitBlockFT0>
{
 public:
  typedef DigitBlockBase<DigitBlockFT0> DigitBlockBaseType;
  DigitBlockFT0(o2::InteractionRecord intRec) { setIntRec(intRec); }
  DigitBlockFT0() = default;
  DigitBlockFT0(const DigitBlockFT0& other) = default;
  ~DigitBlockFT0() = default;
  void setIntRec(o2::InteractionRecord intRec) { mDigit.mIntRecord = intRec; }
  Digit mDigit;
  std::vector<ChannelData> mVecChannelData;
  static int sEventID;

  template <class DataBlockType>
  void processDigits(DataBlockType& dataBlock, int linkID)
  {
    if constexpr (std::is_same<DataBlockType, DataBlockPM>::value) { //Filling data from PM
      for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<RawDataPM>::mNelements; iEventData++) {
        mVecChannelData.emplace_back(uint8_t(o2::ft0::SingleLUT::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].channelID)),
                                     int(dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].time),
                                     int(dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].charge),
                                     dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].getFlagWord());
      }
    } else if constexpr (std::is_same<DataBlockType, DataBlockTCM>::value) { //Filling data from TCM (normal/extended mode)
      dataBlock.DataBlockWrapper<RawDataTCM>::mData[0].fillTrigger(mDigit.mTriggers);
    }
  }
  void getDigits(std::vector<Digit>& vecDigits, std::vector<ChannelData>& vecChannelData)
  {
    //last digit filling
    mDigit.ref.set(vecChannelData.size(), mVecChannelData.size());
    mDigit.mEventID = sEventID;
    //
    vecDigits.push_back(std::move(mDigit));
    std::move(mVecChannelData.begin(), mVecChannelData.end(), std::back_inserter(vecChannelData));
    mVecChannelData.clear();

    sEventID++; //Increasing static eventID. After each poping of the data, it will increase
  }
  void print() const
  {
    std::cout << "\n______________DIGIT DATA____________";
    std::cout << std::hex;
    std::cout << "\nBC: " << mDigit.mIntRecord.bc << "| ORBIT: " << mDigit.mIntRecord.orbit;
    std::cout << "\nRef first: " << mDigit.ref.getFirstEntry() << "| Ref entries: " << mDigit.ref.getEntries();
    std::cout << "\nmTrigger: " << static_cast<uint16_t>(mDigit.mTriggers.triggersignals);
    std::cout << "\nnChanA: " << static_cast<uint16_t>(mDigit.mTriggers.nChanA) << " | nChanC: " << static_cast<uint16_t>(mDigit.mTriggers.nChanC);
    std::cout << "\namplA: " << mDigit.mTriggers.amplA << " | amplC: " << mDigit.mTriggers.amplC;
    std::cout << "\ntimeA: " << mDigit.mTriggers.timeA << " | timeC: " << mDigit.mTriggers.timeC;

    std::cout << "\n______________CHANNEL DATA____________\n";
    std::cout << "\nN channel: " << mVecChannelData.size();
    for (const auto& chData : mVecChannelData) {
      std::cout << "\nChId: " << static_cast<uint16_t>(chData.ChId) << " |  ChainQTC:" << static_cast<uint16_t>(chData.ChainQTC) << " | CFDTime: " << chData.CFDTime << " | QTCAmpl: " << chData.QTCAmpl;
    }
    std::cout << std::dec;
    std::cout << "\n";
    LOG(INFO) << "______________________________________";
  }

  static void print(std::vector<Digit>& vecDigit, std::vector<ChannelData>& vecChannelData)
  {
    for (const auto& digit : vecDigit) {
      std::cout << "\n______________DIGIT DATA____________";
      std::cout << std::hex;
      std::cout << "\nBC: " << digit.mIntRecord.bc << "| ORBIT: " << digit.mIntRecord.orbit << " | EventID: " << digit.mEventID;
      std::cout << "\nRef first: " << digit.ref.getFirstEntry() << "| Ref entries: " << digit.ref.getEntries();
      std::cout << "\nmTrigger: " << static_cast<uint16_t>(digit.mTriggers.triggersignals);
      std::cout << "\nnChanA: " << static_cast<uint16_t>(digit.mTriggers.nChanA) << " | nChanC: " << static_cast<uint16_t>(digit.mTriggers.nChanC);
      std::cout << "\namplA: " << digit.mTriggers.amplA << " | amplC: " << digit.mTriggers.amplC;
      std::cout << "\ntimeA: " << digit.mTriggers.timeA << " | timeC: " << digit.mTriggers.timeC;

      std::cout << "\n______________CHANNEL DATA____________\n";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {

        std::cout << "\nChId: " << static_cast<uint16_t>(vecChannelData[iChData].ChId) << " |  ChainQTC:" << static_cast<uint16_t>(vecChannelData[iChData].ChainQTC)
                  << " | CFDTime: " << vecChannelData[iChData].CFDTime << " | QTCAmpl: " << vecChannelData[iChData].QTCAmpl;
      }
      std::cout << std::dec;
      std::cout << "\n______________________________________\n";
    }
  }
};

//TCM extended data taking mode
class DigitBlockFT0ext : public DigitBlockBase<DigitBlockFT0ext>
{
 public:
  typedef DigitBlockBase<DigitBlockFT0ext> DigitBlockBaseType;

  DigitBlockFT0ext(o2::InteractionRecord intRec) { setIntRec(intRec); }
  DigitBlockFT0ext() = default;
  DigitBlockFT0ext(const DigitBlockFT0ext& other) = default;
  ~DigitBlockFT0ext() = default;
  void setIntRec(o2::InteractionRecord intRec) { mDigit.mIntRecord = intRec; }
  DigitExt mDigit;
  std::vector<ChannelData> mVecChannelData;
  std::vector<TriggersExt> mVecTriggersExt;
  static int sEventID;

  template <class DataBlockType>
  void processDigits(DataBlockType& dataBlock, int linkID)
  {
    if constexpr (std::is_same<DataBlockType, DataBlockPM>::value) { //Filling data from PM
      for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<RawDataPM>::mNelements; iEventData++) {
        mVecChannelData.emplace_back(uint8_t(o2::ft0::SingleLUT::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].channelID)),
                                     int(dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].time),
                                     int(dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].charge),
                                     dataBlock.DataBlockWrapper<RawDataPM>::mData[iEventData].getFlagWord());
      }
    } else if constexpr (std::is_same<DataBlockType, DataBlockTCMext>::value) { //Filling data from TCM, extended mode. Same proccess as for normal mode, for now.
      dataBlock.DataBlockWrapper<RawDataTCM>::mData[0].fillTrigger(mDigit.mTriggers);
      for (int iTriggerWord = 0; iTriggerWord < dataBlock.DataBlockWrapper<RawDataTCMext>::mNelements; iTriggerWord++) {
        mVecTriggersExt.emplace_back(dataBlock.DataBlockWrapper<RawDataTCMext>::mData[iTriggerWord].triggerWord);
      }
    }
  }
  void getDigits(std::vector<DigitExt>& vecDigits, std::vector<ChannelData>& vecChannelData, std::vector<TriggersExt>& vecTriggersExt)
  {
    //last digit filling
    mDigit.ref.set(vecChannelData.size(), mVecChannelData.size());
    mDigit.refExt.set(vecTriggersExt.size(), mVecTriggersExt.size());
    mDigit.mEventID = sEventID;
    //Digits
    vecDigits.push_back(std::move(mDigit));
    //ChannelData
    std::move(mVecChannelData.begin(), mVecChannelData.end(), std::back_inserter(vecChannelData));
    mVecChannelData.clear();
    //TriggersExt
    std::move(mVecTriggersExt.begin(), mVecTriggersExt.end(), std::back_inserter(vecTriggersExt));
    mVecTriggersExt.clear();

    sEventID++; //Increasing static eventID. After each poping of the data, it will increase
  }
  void print() const
  {
    std::cout << "\n______________DIGIT DATA____________";
    std::cout << std::hex;
    std::cout << "\nBC: " << mDigit.mIntRecord.bc << "| ORBIT: " << mDigit.mIntRecord.orbit;
    std::cout << "\nRef first: " << mDigit.ref.getFirstEntry() << "| Ref entries: " << mDigit.ref.getEntries();
    std::cout << "\nmTrigger: " << static_cast<uint16_t>(mDigit.mTriggers.triggersignals);
    std::cout << "\nnChanA: " << static_cast<uint16_t>(mDigit.mTriggers.nChanA) << " | nChanC: " << static_cast<uint16_t>(mDigit.mTriggers.nChanC);
    std::cout << "\namplA: " << mDigit.mTriggers.amplA << " | amplC: " << mDigit.mTriggers.amplC;
    std::cout << "\ntimeA: " << mDigit.mTriggers.timeA << " | timeC: " << mDigit.mTriggers.timeC;

    std::cout << "\n______________CHANNEL DATA____________\n";
    std::cout << "\nN channel: " << mVecChannelData.size();
    for (const auto& chData : mVecChannelData) {
      std::cout << "\nChId: " << static_cast<uint16_t>(chData.ChId) << " |  ChainQTC:" << static_cast<uint16_t>(chData.ChainQTC) << " | CFDTime: " << chData.CFDTime << " | QTCAmpl: " << chData.QTCAmpl;
    }
    std::cout << std::dec;
    std::cout << "\n";
    LOG(INFO) << "______________________________________";
  }

  static void print(std::vector<DigitExt>& vecDigit, std::vector<ChannelData>& vecChannelData, std::vector<TriggersExt>& vecTriggersExt)
  {
    for (const auto& digit : vecDigit) {
      std::cout << "\n______________DIGIT DATA____________";
      std::cout << std::hex;
      std::cout << "\nBC: " << digit.mIntRecord.bc << "| ORBIT: " << digit.mIntRecord.orbit << " | EventID: " << digit.mEventID;
      std::cout << "\nRef first: " << digit.ref.getFirstEntry() << "| Ref entries: " << digit.ref.getEntries();
      std::cout << "\nmTrigger: " << static_cast<uint16_t>(digit.mTriggers.triggersignals);
      std::cout << "\nnChanA: " << static_cast<uint16_t>(digit.mTriggers.nChanA) << " | nChanC: " << static_cast<uint16_t>(digit.mTriggers.nChanC);
      std::cout << "\namplA: " << digit.mTriggers.amplA << " | amplC: " << digit.mTriggers.amplC;
      std::cout << "\ntimeA: " << digit.mTriggers.timeA << " | timeC: " << digit.mTriggers.timeC;

      std::cout << "\n______________CHANNEL DATA____________\n";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {

        std::cout << "\nChId: " << static_cast<uint16_t>(vecChannelData[iChData].ChId) << " |  ChainQTC:" << static_cast<uint16_t>(vecChannelData[iChData].ChainQTC)
                  << " | CFDTime: " << vecChannelData[iChData].CFDTime << " | QTCAmpl: " << vecChannelData[iChData].QTCAmpl;
      }
      std::cout << std::dec;
      std::cout << "\n______________________________________\n";
    }
  }
};

} // namespace ft0
} // namespace o2
#endif
