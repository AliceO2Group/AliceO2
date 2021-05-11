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
//file DigitBlockFIT.h class  for proccessing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKFIT_H_
#define ALICEO2_FIT_DIGITBLOCKFIT_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <Rtypes.h>
#include "FITRaw/DataBlockFIT.h"
#include "FITRaw/DigitBlockBase.h"

#include <CommonDataFormat/InteractionRecord.h>

#include <gsl/span>

namespace o2
{
namespace fit
{
//Normal data taking mode
template <typename LookupTableType, typename Digit, typename ChannelData>
class DigitBlockFIT : public DigitBlockBase<DigitBlockFIT<LookupTableType, Digit, ChannelData>, Digit, ChannelData>
{
 public:
  typedef DigitBlockBase<DigitBlockFIT<LookupTableType, Digit, ChannelData>, Digit, ChannelData> DigitBlockBase_t;
  typedef LookupTableType LookupTable_t;
  template <typename... Args>
  DigitBlockFIT(Args&&... args) : DigitBlockBase_t(std::forward<Args>(args)...)
  {
  }
  DigitBlockFIT() = default;
  DigitBlockFIT(const DigitBlockFIT& other) = default;
  ~DigitBlockFIT() = default;
  //Filling data from PM
  //Temporary for FT0 and FDD
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && !DigitBlockHelper::IsFV0<Digit>::value //Will compile only for FT0 and FDD
                                                                                       >
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      DigitBlockBase_t::mSubDigit.emplace_back(static_cast<uint8_t>(LookupTable_t::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].channelID, ep)),
                                               static_cast<int>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].time),
                                               static_cast<int>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].charge),
                                               static_cast<uint8_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].getFlagWord()));
    }
  }
  //
  //Temporary for FV0
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && DigitBlockHelper::IsFV0<Digit>::value //Will compile only for FV0
                                                                                       >
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      DigitBlockBase_t::mSubDigit.emplace_back(static_cast<Short_t>(LookupTable_t::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].channelID)),
                                               static_cast<Float_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].time),
                                               static_cast<Short_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].charge)
                                               /*,dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].getFlagWord()*/);
    }
  }

  //Filling data from TCM (normal mode)
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCM, DataBlockType>::value>
  {
    dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0].fillTrigger(DigitBlockBase_t::mDigit.mTriggers);
  }
  template <typename DetTrigInput>
  void getDigits(std::vector<Digit>& vecDigits, std::vector<ChannelData>& vecChannelData, std::vector<DetTrigInput>& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  void getDigits(std::vector<Digit>& vecDigits, std::vector<ChannelData>& vecChannelData)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  static void print(const std::vector<Digit>& vecDigit, const std::vector<ChannelData>& vecChannelData)
  {
    for (const auto& digit : vecDigit) {
      digit.printLog();
      LOG(INFO) << "______________CHANNEL DATA____________";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {
        vecChannelData[iChData].printLog();
      }
      LOG(INFO) << "______________________________________";
    }
  }
};

//TCM extended data taking mode
template <typename LookupTableType, typename Digit, typename ChannelData, typename TriggersExt>
class DigitBlockFIText : public DigitBlockBase<DigitBlockFIText<LookupTableType, Digit, ChannelData, TriggersExt>, Digit, ChannelData, TriggersExt>
{
 public:
  typedef DigitBlockBase<DigitBlockFIText<LookupTableType, Digit, ChannelData, TriggersExt>, Digit, ChannelData, TriggersExt> DigitBlockBase_t;
  typedef LookupTableType LookupTable_t;
  template <typename... Args>
  DigitBlockFIText(Args&&... args) : DigitBlockBase_t(std::forward<Args>(args)...)
  {
  }
  DigitBlockFIText() = default;
  DigitBlockFIText(const DigitBlockFIText& other) = default;
  ~DigitBlockFIText() = default;
  //Filling data from PM
  //Temporary for FT0 and FDD
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && !DigitBlockHelper::IsFV0<Digit>::value //Will compile only for FT0 and FDD
                                                                                       >
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      DigitBlockBase_t::mSubDigit.emplace_back(static_cast<uint8_t>(LookupTable_t::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].channelID, ep)),
                                               static_cast<int>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].time),
                                               static_cast<int>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].charge),
                                               static_cast<uint8_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].getFlagWord()));
    }
  }
  //
  //Temporary for FV0
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && DigitBlockHelper::IsFV0<Digit>::value //Will compile only for FV0
                                                                                       >
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      DigitBlockBase_t::mSubDigit.emplace_back(static_cast<Short_t>(LookupTable_t::Instance().getChannel(linkID, dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].channelID)),
                                               static_cast<Float_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].time),
                                               static_cast<Short_t>(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].charge)
                                               /*,dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData].getFlagWord()*/);
    }
  }
  //Filling data from TCM (extended mode)
  //Temporary for FT0 and FDD
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value && !DigitBlockHelper::IsFV0<Digit>::value>
  {
    dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0].fillTrigger(DigitBlockBase_t::mDigit.mTriggers);
    DigitBlockBase_t::mSingleSubDigit.mIntRecord = DigitBlockBase_t::mDigit.mIntRecord;
    for (int iTriggerWord = 0; iTriggerWord < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mNelements; iTriggerWord++) {
      DigitBlockBase_t::mSingleSubDigit.setTrgWord(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[iTriggerWord].triggerWord, iTriggerWord);
    }
  }

  //Temporary for FV0
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value && DigitBlockHelper::IsFV0<Digit>::value> //Will compile only for FV0
  {
    dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0].fillTrigger(DigitBlockBase_t::mDigit.mTriggers);
    DigitBlockBase_t::mSingleSubDigit.mIntRecord = DigitBlockBase_t::mDigit.ir;
    for (int iTriggerWord = 0; iTriggerWord < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mNelements; iTriggerWord++) {
      DigitBlockBase_t::mSingleSubDigit.setTrgWord(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[iTriggerWord].triggerWord, iTriggerWord);
    }
  }
  template <typename DetTrigInput>
  void getDigits(std::vector<Digit>& vecDigits, std::vector<ChannelData>& vecChannelData, std::vector<TriggersExt>& vecTriggersExt, std::vector<DetTrigInput>& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    getDigits(vecDigits, vecChannelData, vecTriggersExt);
  }

  void getDigits(std::vector<Digit>& vecDigits, std::vector<ChannelData>& vecChannelData, std::vector<TriggersExt>& vecTriggersExt)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
    DigitBlockBase_t::getSingleSubDigits(vecTriggersExt);
  }
  static void print(const std::vector<Digit>& vecDigit, const std::vector<ChannelData>& vecChannelData, const std::vector<TriggersExt>& vecTriggersExt)
  {
    for (const auto& digit : vecDigit) {
      digit.printLog();
      LOG(INFO) << "______________CHANNEL DATA____________";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {
        vecChannelData[iChData].printLog();
      }
      LOG(INFO) << "______________________________________";
    }
    LOG(INFO) << "______________EXTENDED TRIGGERS____________";
    for (const auto& trgExt : vecTriggersExt) {
      trgExt.printLog();
    }
    LOG(INFO) << "______________________________________";
  }
};

} // namespace fit
} // namespace o2
#endif
