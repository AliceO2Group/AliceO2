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

#include "TTree.h"

#include <gsl/span>

namespace o2
{
namespace fit
{
//Temporary helper
namespace DigitBlockFIThelper
{
template <typename T, typename = void>
struct IsFV0;
//FV0 Digit recognition
template <typename T>
struct IsFV0<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().mIntRecord), o2::InteractionRecord>::value>> : std::false_type {
};
template <typename T>
struct IsFV0<T, std::enable_if_t<std::is_same<decltype(std::declval<T>().ir), o2::InteractionRecord>::value>> : std::true_type {
};
//Temporary, PM module convertation
//FT0
template <typename ChannelDataType, typename PMDataType>
auto ConvertChData2EventData(const ChannelDataType& chData, PMDataType& pmData, int channelID) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().QTCAmpl), int16_t>::value>
{
  pmData.word = uint64_t(chData.ChainQTC) << PMDataType::BitFlagPos;
  pmData.channelID = channelID;
  pmData.time = chData.CFDTime;
  pmData.charge = chData.QTCAmpl;
}
//FV0
template <typename ChannelDataType, typename PMDataType>
auto ConvertChData2EventData(const ChannelDataType& chData, PMDataType& pmData, int channelID) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().chargeAdc), Short_t>::value>
{
  pmData.channelID = channelID;
  pmData.time = chData.time;
  pmData.charge = chData.chargeAdc;
}
//FDD
template <typename ChannelDataType, typename PMDataType>
auto ConvertChData2EventData(const ChannelDataType& chData, PMDataType& pmData, int channelID) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().mChargeADC), int16_t>::value>
{
  pmData.word = uint64_t(chData.mFEEBits) << PMDataType::BitFlagPos;
  pmData.channelID = channelID;
  pmData.time = chData.mTime;
  pmData.charge = chData.mChargeADC;
}
//Temporary, TCM module convertation
//FT0 and FDD
template <typename DigitType, typename TCMDataType>
auto ConvertDigit2TCMData(const DigitType& digit, TCMDataType& tcmData) -> std::enable_if_t<!IsFV0<DigitType>::value>
{
  tcmData.orA = digit.mTriggers.getOrA();
  tcmData.orC = digit.mTriggers.getOrC();
  tcmData.sCen = digit.mTriggers.getVertex();
  tcmData.cen = digit.mTriggers.getCen();
  tcmData.vertex = digit.mTriggers.getSCen();
  tcmData.laser = 0;
  //tcmData.laser = digit.mTriggers.getLaserBit(); //Turned off for FDD
  tcmData.nChanA = digit.mTriggers.nChanA;
  tcmData.nChanC = digit.mTriggers.nChanC;
  if (digit.mTriggers.amplA > 131071) {
    tcmData.amplA = 131071; //2^17
  } else {
    tcmData.amplA = digit.mTriggers.amplA;
  }
  if (digit.mTriggers.amplC > 131071) {
    tcmData.amplC = 131071; //2^17
  } else {
    tcmData.amplC = digit.mTriggers.amplC;
  }
  tcmData.timeA = digit.mTriggers.timeA;
  tcmData.timeC = digit.mTriggers.timeC;
}
//FV0
template <typename DigitType, typename TCMDataType>
auto ConvertDigit2TCMData(const DigitType& digit, TCMDataType& tcmData) -> std::enable_if_t<IsFV0<DigitType>::value>
{
  tcmData.orA = bool(digit.mTriggers.triggerSignals & (1 << 0));
  tcmData.orC = bool(digit.mTriggers.triggerSignals & (1 << 1));
  tcmData.sCen = bool(digit.mTriggers.triggerSignals & (1 << 2));
  tcmData.cen = bool(digit.mTriggers.triggerSignals & (1 << 3));
  tcmData.vertex = bool(digit.mTriggers.triggerSignals & (1 << 4));
  tcmData.laser = bool(digit.mTriggers.triggerSignals & (1 << 5));
  tcmData.nChanA = digit.mTriggers.nChanA;
  //tcmData.nChanC = digit.mTriggers.nChanC;
  tcmData.nChanC = 0;
  tcmData.amplA = digit.mTriggers.amplA;
  //tcmdata.amplC = digit.mTriggers.amplA;
  tcmData.amplC = 0;
  //tcmData.timeA = digit.mTriggers.timeA
  //tcmData.timeC = digit.mTriggers.timeC;
  tcmData.timeA = 0;
  tcmData.timeC = 0;
}

template <typename T>
auto GetIntRecord(const T& digit) -> std::enable_if_t<!IsFV0<T>::value, o2::InteractionRecord>
{
  return digit.mIntRecord;
}
template <typename T>
auto GetIntRecord(const T& digit) -> std::enable_if_t<IsFV0<T>::value, o2::InteractionRecord>
{
  return digit.ir;
}
} // namespace DigitBlockFIThelper

//Normal data taking mode
template <typename LookupTableType, typename DigitType, typename ChannelDataType>
class DigitBlockFIT : public DigitBlockBase<DigitType, ChannelDataType>
{
 public:
  using DigitBlockFIT_t = DigitBlockFIT<LookupTableType, DigitType, ChannelDataType>;
  typedef DigitBlockBase<DigitType, ChannelDataType> DigitBlockBase_t;
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
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && !DigitBlockFIThelper::IsFV0<DigitType>::value //Will compile only for FT0 and FDD
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
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && DigitBlockFIThelper::IsFV0<DigitType>::value //Will compile only for FV0
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
  //Decompose digits into DataBlocks
  //DataBlockPM
  template <class DataBlockType>
  auto decomposeDigits() const -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value, std::map<typename LookupTable_t::Topo_t, DataBlockType>>
  {
    using Topo_t = typename LookupTable_t::Topo_t;
    std::map<Topo_t, DataBlockType> mapResult;
    std::map<Topo_t, std::reference_wrapper<const ChannelDataType>> mapTopo2SortedCh;
    std::map<Topo_t, std::size_t> mapTopoCounter;
    //Preparing map "Topo to ChannelData refs" and map "Global Topo(FEE metadata) to number of ChannelData"
    for (const auto& entry : DigitBlockBase_t::mSubDigit) {
      auto topoPM = LookupTable_t::Instance().getTopoPM(static_cast<int>(entry.getChannelID()));
      mapTopo2SortedCh.insert({topoPM, entry});
      auto pairInserted = mapTopoCounter.insert({LookupTable_t::makeGlobalTopo(topoPM), 0});
      pairInserted.first->second++;
    }
    //Preparing map of global Topo(related to PM module) to DataBlockPMs
    for (const auto& entry : mapTopo2SortedCh) {
      auto pairInserted = mapResult.insert({LookupTable_t::makeGlobalTopo(entry.first), {}});
      auto& refDataBlock = pairInserted.first->second;
      if (pairInserted.second) {
        //Header preparation
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].setIntRec(DigitBlockFIThelper::GetIntRecord(DigitBlockBase_t::mDigit));
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].startDescriptor = 0xf;
        std::size_t nElements = mapTopoCounter.find(pairInserted.first->first)->second;
        std::size_t nWords = nElements / 2 + nElements % 2;
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].nGBTWords = nWords;
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mNelements = 1;
      }
      //Data preparation
      auto& refPos = refDataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements;
      auto& refData = refDataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[refPos];
      refPos++;
      DigitBlockFIThelper::ConvertChData2EventData(entry.second.get(), refData, LookupTable_t::Instance().getLocalChannelID(entry.first));
    }
    return mapResult;
  }
  //DataBlockTCM
  template <class DataBlockType>
  auto decomposeDigits() const -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCM, DataBlockType>::value, std::pair<typename LookupTable_t::Topo_t, DataBlockType>>
  {
    DataBlockType dataBlockTCM{};
    //Header preparation
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCM>::mData[0].setIntRec(DigitBlockFIThelper::GetIntRecord(DigitBlockBase_t::mDigit));
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCM>::mData[0].startDescriptor = 0xf;
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCM>::mData[0].nGBTWords =
      dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::MaxNwords;
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCM>::mNelements = 1;
    auto& refTCMdata = dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0];
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mNelements = 1;

    //Data preparation
    DigitBlockFIThelper::ConvertDigit2TCMData(DigitBlockBase_t::mDigit, refTCMdata);
    return {LookupTable_t::Instance().getTopoTCM(), dataBlockTCM};
  }
  //Process DigitBlocks from TTree
  template <typename DigitBlockProcType>
  static void processDigitBlocks(TTree* inputTree, DigitBlockProcType& digitBlockProc)
  {
    assert(inputTree != nullptr);
    std::vector<DigitBlockFIT_t> vecResult;
    std::vector<typename DigitBlockBase_t::Digit_t> vecDigit;
    std::vector<typename DigitBlockBase_t::Digit_t>* ptrVecDigit = &vecDigit;
    typename DigitBlockBase_t::SubDigit_t vecChannelData;
    typename DigitBlockBase_t::SubDigit_t* ptrVecChannelData = &vecChannelData;
    inputTree->SetBranchAddress(decltype(vecDigit)::value_type::sDigitBranchName, &ptrVecDigit);
    inputTree->SetBranchAddress(decltype(vecChannelData)::value_type::sDigitBranchName, &ptrVecChannelData);
    for (int iEntry = 0; iEntry < inputTree->GetEntries(); iEntry++) {
      inputTree->GetEntry(iEntry);
      LOG(INFO) << "Processing TF " << iEntry;
      digitBlockProc.processDigitBlockPerTF(DigitBlockBase_t::template makeDigitBlock<DigitBlockFIT_t>(vecDigit, vecChannelData));
    }
  }
  //
  template <typename DetTrigInput>
  void getDigits(std::vector<DigitType>& vecDigits, std::vector<ChannelDataType>& vecChannelData, std::vector<DetTrigInput>& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  void getDigits(std::vector<DigitType>& vecDigits, std::vector<ChannelDataType>& vecChannelData)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  static void print(const std::vector<DigitType>& vecDigit, const std::vector<ChannelDataType>& vecChannelData)
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
  void print() const
  {
    DigitBlockBase_t::print();
  }
};

//TCM extended data taking mode
template <typename LookupTableType, typename DigitType, typename ChannelDataType, typename TriggersExtType>
class DigitBlockFIText : public DigitBlockBase<DigitType, ChannelDataType, TriggersExtType>
{
 public:
  using DigitBlockFIT_t = DigitBlockFIText<LookupTableType, DigitType, ChannelDataType, TriggersExtType>;
  typedef DigitBlockBase<DigitType, ChannelDataType, TriggersExtType> DigitBlockBase_t;
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
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && !DigitBlockFIThelper::IsFV0<DigitType>::value //Will compile only for FT0 and FDD
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
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value && DigitBlockFIThelper::IsFV0<DigitType>::value //Will compile only for FV0
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
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value && !DigitBlockFIThelper::IsFV0<DigitType>::value>
  {
    dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0].fillTrigger(DigitBlockBase_t::mDigit.mTriggers);
    DigitBlockBase_t::mSingleSubDigit.mIntRecord = DigitBlockBase_t::mDigit.mIntRecord;
    for (int iTriggerWord = 0; iTriggerWord < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mNelements; iTriggerWord++) {
      DigitBlockBase_t::mSingleSubDigit.setTrgWord(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[iTriggerWord].triggerWord, iTriggerWord);
    }
  }

  //Temporary for FV0
  template <class DataBlockType>
  auto processDigits(DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value && DigitBlockFIThelper::IsFV0<DigitType>::value> //Will compile only for FV0
  {
    dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0].fillTrigger(DigitBlockBase_t::mDigit.mTriggers);
    DigitBlockBase_t::mSingleSubDigit.mIntRecord = DigitBlockBase_t::mDigit.ir;
    for (int iTriggerWord = 0; iTriggerWord < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mNelements; iTriggerWord++) {
      DigitBlockBase_t::mSingleSubDigit.setTrgWord(dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[iTriggerWord].triggerWord, iTriggerWord);
    }
  }

  //Decompose digits into DataBlocks
  //DataBlockPM
  template <class DataBlockType>
  auto decomposeDigits() const -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value, std::map<typename LookupTable_t::Topo_t, DataBlockType>>
  {
    using Topo_t = typename LookupTable_t::Topo_t;
    std::map<Topo_t, DataBlockType> mapResult;
    std::map<Topo_t, std::reference_wrapper<const ChannelDataType>> mapTopo2SortedCh;
    std::map<Topo_t, std::size_t> mapTopoCounter;
    //Preparing map "Topo to ChannelData refs" and map "Global Topo(FEE metadata) to number of ChannelData"
    for (const auto& entry : DigitBlockBase_t::mSubDigit) {
      auto topoPM = LookupTable_t::Instance().getTopoPM(static_cast<int>(entry.getChannelID()));
      mapTopo2SortedCh.insert({topoPM, entry});
      auto pairInserted = mapTopoCounter.insert({LookupTable_t::makeGlobalTopo(topoPM), 0});
      pairInserted.first->second++;
    }
    //Preparing map of global Topo(related to PM module) to DataBlockPMs
    for (const auto& entry : mapTopo2SortedCh) {
      auto pairInserted = mapResult.insert({LookupTable_t::makeGlobalTopo(entry.first), {}});
      auto& refDataBlock = pairInserted.first->second;
      if (pairInserted.second) {
        //Header preparation
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].setIntRec(DigitBlockFIThelper::GetIntRecord(DigitBlockBase_t::mDigit));
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].startDescriptor = 0xf;
        std::size_t nElements = mapTopoCounter.find(pairInserted.first->first)->second;
        std::size_t nWords = nElements / 2 + nElements % 2;
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mData[0].nGBTWords = nWords;
        refDataBlock.DataBlockWrapper<typename DataBlockType::RawHeaderPM>::mNelements = 1;
      }
      //Data preparation
      auto& refPos = refDataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements;
      auto& refData = refDataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[refPos];
      refPos++;
      DigitBlockFIThelper::ConvertChData2EventData(entry.second.get(), refData, LookupTable_t::Instance().getLocalChannelID(entry.first));
    }
    return mapResult;
  }
  //DataBlockTCM
  template <class DataBlockType>
  auto decomposeDigits() const -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value, std::pair<typename LookupTable_t::Topo_t, DataBlockType>>
  {
    DataBlockType dataBlockTCM{};
    //Header preparation
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCMext>::mData[0].setIntRec(DigitBlockFIThelper::GetIntRecord(DigitBlockBase_t::mDigit));
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCMext>::mData[0].startDescriptor = 0xf;
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCMext>::mData[0].nGBTWords = dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::MaxNwords +
                                                                                                 dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::MaxNwords;

    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawHeaderTCM>::mNelements = 1;
    auto& refTCMdata = dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0];
    //Data preparation
    DigitBlockFIThelper::ConvertDigit2TCMData(DigitBlockBase_t::mDigit, refTCMdata);
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mNelements = 1;
    //Extended mode
    static_assert(std::decay<decltype(dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[0])>::type::MaxNelements == std::tuple_size<decltype(DigitBlockBase_t::mSingleSubDigit.mTriggerWords)>::value);
    for (int i = 0; i < std::decay<decltype(dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[0])>::type::MaxNelements; i++) {
      dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mData[i].triggerWord = DigitBlockBase_t::mSingleSubDigit.mTriggerWords[i];
    }
    dataBlockTCM.DataBlockWrapper<typename DataBlockType::RawDataTCMext>::mNelements = 1;
    return {LookupTable_t::Instance().getTopoTCM(), dataBlockTCM};
  }
  template <typename DetTrigInput>
  void getDigits(std::vector<DigitType>& vecDigits, std::vector<ChannelDataType>& vecChannelData, std::vector<TriggersExtType>& vecTriggersExt, std::vector<DetTrigInput>& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    getDigits(vecDigits, vecChannelData, vecTriggersExt);
  }

  void getDigits(std::vector<DigitType>& vecDigits, std::vector<ChannelDataType>& vecChannelData, std::vector<TriggersExtType>& vecTriggersExt)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
    DigitBlockBase_t::getSingleSubDigits(vecTriggersExt);
  }
  static void print(const std::vector<DigitType>& vecDigit, const std::vector<ChannelDataType>& vecChannelData, const std::vector<TriggersExtType>& vecTriggersExt)
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
  void print() const
  {
    DigitBlockBase_t::print();
  }
};
} // namespace fit
} // namespace o2
#endif
