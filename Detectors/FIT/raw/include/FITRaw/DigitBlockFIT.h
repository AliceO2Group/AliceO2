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

// Temporary, PM module conversion
// FT0 & FV0
template <typename ChannelDataType, typename PMDataType>
auto ConvertChData2EventData(const ChannelDataType& chData, PMDataType& pmData, int channelID) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().QTCAmpl), int16_t>::value>
{
  pmData.word = uint64_t(chData.ChainQTC) << PMDataType::BitFlagPos;
  pmData.channelID = channelID;
  pmData.time = chData.CFDTime;
  pmData.charge = chData.QTCAmpl;
}
// FDD
template <typename ChannelDataType, typename PMDataType>
auto ConvertChData2EventData(const ChannelDataType& chData, PMDataType& pmData, int channelID) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().mChargeADC), int16_t>::value>
{
  pmData.word = uint64_t(chData.mFEEBits) << PMDataType::BitFlagPos;
  pmData.channelID = channelID;
  pmData.time = chData.mTime;
  pmData.charge = chData.mChargeADC;
}

// Temporary, TCM module conversion
// FT0, FV0 and FDD
template <typename DigitType, typename TCMDataType>
auto ConvertDigit2TCMData(const DigitType& digit, TCMDataType& tcmData)
{
  tcmData.orA = digit.mTriggers.getOrA();
  tcmData.orC = digit.mTriggers.getOrC();
  tcmData.sCen = digit.mTriggers.getSCen();
  tcmData.cen = digit.mTriggers.getCen();
  tcmData.vertex = digit.mTriggers.getVertex();
  tcmData.laser = digit.mTriggers.getLaser();
  tcmData.outputsAreBlocked = digit.mTriggers.getOutputsAreBlocked();
  tcmData.dataIsValid = digit.mTriggers.getDataIsValid();
  tcmData.nChanA = digit.mTriggers.getNChanA();
  tcmData.nChanC = digit.mTriggers.getNChanC();
  const int64_t thresholdSignedInt17bit = 65535; //pow(2,17)/2-1
  if (digit.mTriggers.getAmplA() > thresholdSignedInt17bit) {
    tcmData.amplA = thresholdSignedInt17bit;
  } else {
    tcmData.amplA = digit.mTriggers.getAmplA();
  }
  if (digit.mTriggers.getAmplC() > thresholdSignedInt17bit) {
    tcmData.amplC = thresholdSignedInt17bit;
  } else {
    tcmData.amplC = digit.mTriggers.getAmplC();
  }
  tcmData.timeA = digit.mTriggers.getTimeA();
  tcmData.timeC = digit.mTriggers.getTimeC();
}

// Digit to raw helper functions, temporary
// TCM to Digit convertation
// FT0, FV0 and FDD
template <typename DigitType, typename TCMDataType>
auto ConvertTCMData2Digit(DigitType& digit, const TCMDataType& tcmData)
{
  using TriggerType = decltype(digit.mTriggers);
  auto& trg = digit.mTriggers;
  trg.setTriggers((bool)tcmData.orA, (bool)tcmData.orC, (bool)tcmData.vertex, (bool)tcmData.cen, (bool)tcmData.sCen,
                  (int8_t)tcmData.nChanA, (int8_t)tcmData.nChanC, (int32_t)tcmData.amplA, (int32_t)tcmData.amplC,
                  (int16_t)tcmData.timeA, (int16_t)tcmData.timeC, (bool)tcmData.laser, (bool)tcmData.outputsAreBlocked, (bool)tcmData.dataIsValid);
}

// PM to ChannelData convertation
// FT0 and FV0
template <typename LookupTableType, typename ChannelDataType, typename PMDataType>
auto ConvertEventData2ChData(std::vector<ChannelDataType>& vecChData, const PMDataType& pmData, int linkID, int ep) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().QTCAmpl), int16_t>::value>
{
  bool isValid{};
  const auto globalChID = LookupTableType::Instance().getChannel(linkID, ep, pmData.channelID, isValid);
  if (isValid) {
    vecChData.emplace_back(static_cast<uint8_t>(globalChID), static_cast<int>(pmData.time), static_cast<int>(pmData.charge), static_cast<uint8_t>(pmData.getFlagWord()));
  } else {
    static int warningCount = 0;
    if (warningCount++ < 100) {
      LOG(warning) << "Incorrect global channel! linkID: " << linkID << " | EndPoint: " << ep << " | LocalChID: " << pmData.channelID;
    }
  }
}
// FDD
template <typename LookupTableType, typename ChannelDataType, typename PMDataType>
auto ConvertEventData2ChData(std::vector<ChannelDataType>& vecChData, const PMDataType& pmData, int linkID, int ep) -> std::enable_if_t<std::is_same<decltype(std::declval<ChannelDataType>().mChargeADC), int16_t>::value>
{
  bool isValid{};
  const auto globalChID = LookupTableType::Instance().getChannel(linkID, ep, pmData.channelID, isValid);
  if (isValid) {
    vecChData.emplace_back(static_cast<uint8_t>(globalChID), static_cast<int>(pmData.time), static_cast<int>(pmData.charge), static_cast<uint8_t>(pmData.getFlagWord()));
  } else {
    static int warningCount = 0;
    if (warningCount++ < 100) {
      LOG(warning) << "Incorrect global channel! linkID: " << linkID << " | EndPoint: " << ep << " | LocalChID: " << pmData.channelID;
    }
  }
}
//Interface for extracting interaction record from Digit
template <typename T>
auto GetIntRecord(const T& digit)
{
  return digit.mIntRecord;
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
  template <class DataBlockType>
  auto processDigits(const DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value>
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      const auto& pmData = dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData];
      DigitBlockFIThelper::ConvertEventData2ChData<LookupTable_t>(DigitBlockBase_t::mSubDigit, pmData, linkID, ep);
    }
  }
  //Filling data from TCM (normal mode)
  template <class DataBlockType>
  auto processDigits(const DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCM, DataBlockType>::value>
  {
    auto& tcmData = dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0];
    DigitBlockFIThelper::ConvertTCMData2Digit(DigitBlockBase_t::mDigit, tcmData);
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
      LOG(info) << "Processing TF " << iEntry;
      digitBlockProc.processDigitBlockPerTF(DigitBlockBase_t::template makeDigitBlock<DigitBlockFIT_t>(vecDigit, vecChannelData));
    }
  }
  template <typename VecDigitType, typename VecChannelDataType, typename VecDetTrigInputType>
  void getDigits(VecDigitType& vecDigits, VecChannelDataType& vecChannelData, VecDetTrigInputType& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  template <typename VecDigitType, typename VecChannelDataType>
  void getDigits(VecDigitType& vecDigits, VecChannelDataType& vecChannelData)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
  }
  static void print(const std::vector<DigitType>& vecDigit, const std::vector<ChannelDataType>& vecChannelData)
  {
    for (const auto& digit : vecDigit) {
      digit.printLog();
      LOG(info) << "______________CHANNEL DATA____________";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {
        vecChannelData[iChData].printLog();
      }
      LOG(info) << "______________________________________";
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
  template <class DataBlockType>
  auto processDigits(const DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockPM, DataBlockType>::value>
  {
    for (int iEventData = 0; iEventData < dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mNelements; iEventData++) {
      const auto& pmData = dataBlock.DataBlockWrapper<typename DataBlockType::RawDataPM>::mData[iEventData];
      DigitBlockFIThelper::ConvertEventData2ChData<LookupTable_t>(DigitBlockBase_t::mSubDigit, pmData, linkID, ep);
    }
  }
  //Filling data from TCM (extended mode)
  template <class DataBlockType>
  auto processDigits(const DataBlockType& dataBlock, int linkID, int ep) -> std::enable_if_t<DigitBlockHelper::IsSpecOfType<DataBlockTCMext, DataBlockType>::value>
  {
    auto& tcmData = dataBlock.DataBlockWrapper<typename DataBlockType::RawDataTCM>::mData[0];
    DigitBlockFIThelper::ConvertTCMData2Digit(DigitBlockBase_t::mDigit, tcmData);
    DigitBlockBase_t::mSingleSubDigit.mIntRecord = DigitBlockFIThelper::GetIntRecord(DigitBlockBase_t::mDigit);
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
  template <typename VecDigitType, typename VecChannelDataType, typename VecTriggersExtType, typename VecDetTrigInputType>
  void getDigits(VecDigitType& vecDigits, VecChannelDataType& vecChannelData, VecTriggersExtType& vecTriggersExt, VecDetTrigInputType& vecTriggerInput)
  {
    DigitBlockBase_t::mDigit.fillTrgInputVec(vecTriggerInput);
    getDigits(vecDigits, vecChannelData, vecTriggersExt);
  }
  template <typename VecDigitType, typename VecChannelDataType, typename VecTriggersExtType>
  void getDigits(VecDigitType& vecDigits, VecChannelDataType& vecChannelData, VecTriggersExtType& vecTriggersExt)
  {
    DigitBlockBase_t::getSubDigits(vecDigits, vecChannelData);
    DigitBlockBase_t::getSingleSubDigits(vecTriggersExt);
  }
  static void print(const std::vector<DigitType>& vecDigit, const std::vector<ChannelDataType>& vecChannelData, const std::vector<TriggersExtType>& vecTriggersExt)
  {
    for (const auto& digit : vecDigit) {
      digit.printLog();
      LOG(info) << "______________CHANNEL DATA____________";
      for (int iChData = digit.ref.getFirstEntry(); iChData < digit.ref.getFirstEntry() + digit.ref.getEntries(); iChData++) {
        vecChannelData[iChData].printLog();
      }
      LOG(info) << "______________________________________";
    }
    LOG(info) << "______________EXTENDED TRIGGERS____________";
    for (const auto& trgExt : vecTriggersExt) {
      trgExt.printLog();
    }
    LOG(info) << "______________________________________";
  }
  void print() const
  {
    DigitBlockBase_t::print();
  }
};
} // namespace fit
} // namespace o2
#endif
