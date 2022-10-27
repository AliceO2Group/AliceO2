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

/// \file TPCDistributeIDCSpec.h
/// \brief TPC aggregation of grouped IDCs and factorization
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 22, 2021

#ifndef O2_TPCDISTRIBUTEIDCIDCSPEC_H
#define O2_TPCDISTRIBUTEIDCIDCSPEC_H

#include <vector>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/TPCFLPIDCSpec.h"
#include "TPCBase/CRU.h"
#include "MemoryResources/MemoryResources.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonDataFormat/Pair.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCDistributeIDCSpec : public o2::framework::Task
{
 public:
  TPCDistributeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const int nTFsBuffer, const unsigned int outlanes, const int firstTF, std::shared_ptr<o2::base::GRPGeomRequest> req)
    : mCRUs{crus}, mTimeFrames{timeframes}, mNTFsBuffer{nTFsBuffer}, mOutLanes{outlanes}, mProcessedCRU{{std::vector<unsigned int>(timeframes), std::vector<unsigned int>(timeframes)}}, mTFStart{{firstTF, firstTF + timeframes}}, mTFEnd{{firstTF + timeframes - 1, mTFStart[1] + timeframes - 1}}, mCCDBRequest(req), mSendCCDBOutput(outlanes)
  {
    // pre calculate data description for output
    mDataDescrOut.reserve(mOutLanes);
    for (unsigned int i = 0; i < mOutLanes; ++i) {
      mDataDescrOut.emplace_back(getDataDescriptionIDC(i));
    }

    // sort vector for binary_search
    std::sort(mCRUs.begin(), mCRUs.end());

    for (auto& processedCRUbuffer : mProcessedCRUs) {
      processedCRUbuffer.resize(mTimeFrames);
      for (auto& crusMap : processedCRUbuffer) {
        crusMap.reserve(mCRUs.size());
        for (const auto cruID : mCRUs) {
          crusMap.emplace(cruID, false);
        }
      }
    }

    const auto sides = IDCFactorization::getSides(mCRUs);
    for (auto side : sides) {
      const std::string name = (side == Side::A) ? "idcsgroupa" : "idcsgroupc";
      mFilter.emplace_back(InputSpec{name.data(), ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPIDCDevice::getDataDescriptionIDCGroup(side)}, Lifetime::Timeframe});
    }
  };

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNFactorTFs = ic.options().get<int>("nFactorTFs");
    mNTFsDataDrop = ic.options().get<int>("drop-data-after-nTFs");
    mCheckEveryNData = ic.options().get<int>("check-data-every-n");
    if (mCheckEveryNData == 0) {
      mCheckEveryNData = mTimeFrames / 2;
      mNTFsDataDrop = mCheckEveryNData;
    }
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    // send data only when object are updated
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      std::fill(mSendCCDBOutput.begin(), mSendCCDBOutput.end(), true);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // send orbit reset and orbits per TF only once
    if (mCCDBRequest->askTime) {
      if (pc.inputs().isValid("grpecs") && pc.inputs().isValid("orbitReset")) {
        o2::base::GRPGeomHelper::instance().checkUpdates(pc);
        if (pc.inputs().countValidInputs() == 2) {
          return;
        }
      }
    }

    const auto tf = processing_helpers::getCurrentTF(pc);

    // automatically detect firstTF in case firstTF was not specified
    if (mTFStart.front() <= -1) {
      const auto firstTF = tf;
      const long offsetTF = std::abs(mTFStart.front() + 1);
      const auto nTotTFs = getNRealTFs();
      mTFStart = {firstTF + offsetTF, firstTF + offsetTF + nTotTFs};
      mTFEnd = {mTFStart[1] - 1, mTFStart[1] - 1 + nTotTFs};
      LOGP(info, "Setting {} as first TF", mTFStart[0]);
      LOGP(info, "Using offset of {} TFs for setting the first TF", offsetTF);
    }

    // check which buffer to use for current incoming data
    const bool currentBuffer = (tf > mTFEnd[mBuffer]) ? !mBuffer : mBuffer;
    if (mTFStart[currentBuffer] > tf) {
      LOGP(info, "all CRUs for current TF {} already received. Skipping this TF", tf);
      return;
    }

    const unsigned int currentOutLane = getOutLane(tf);
    const unsigned int relTF = (tf - mTFStart[currentBuffer]) / mNTFsBuffer;
    LOGP(debug, "current TF: {}   relative TF: {}    current buffer: {}    current output lane: {}     mTFStart: {}", tf, relTF, currentBuffer, currentOutLane, mTFStart[currentBuffer]);

    if (relTF >= mProcessedCRU[currentBuffer].size()) {
      LOGP(warning, "Skipping tf {}: relative tf {} is larger than size of buffer: {}", tf, relTF, mProcessedCRU[currentBuffer].size());

      // check number of processed CRUs for previous TFs. If CRUs are missing for them, they are probably lost/not received
      mProcessedTotalData = mCheckEveryNData;
      checkIntervalsForMissingData(pc, currentBuffer, relTF, currentOutLane, tf);
      return;
    }

    if (mProcessedCRU[currentBuffer][relTF] == mCRUs.size()) {
      return;
    }

    // send start info only once
    if (mSendOutputStartInfo[currentBuffer]) {
      mSendOutputStartInfo[currentBuffer] = false;
      pc.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionIDCFirstTF(), header::DataHeader::SubSpecificationType{currentOutLane}}, mTFStart[currentBuffer]);
    }

    if (mSendCCDBOutput[currentOutLane]) {
      mSendCCDBOutput[currentOutLane] = false;
      pc.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionIDCOrbitReset(), header::DataHeader::SubSpecificationType{currentOutLane}}, dataformats::Pair<long, int>{o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS(), o2::base::GRPGeomHelper::instance().getNHBFPerTF()});
    }

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = tpcCRUHeader->subSpecification >> 7;

      // check if cru is specified in input cru list
      if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
        LOGP(debug, "Received data from CRU: {} which was not specified as input. Skipping", cru);
        continue;
      }

      if (mProcessedCRUs[currentBuffer][relTF][cru]) {
        continue;
      } else {
        // count total number of processed CRUs for given TF
        ++mProcessedCRU[currentBuffer][relTF];

        // to keep track of processed CRUs
        mProcessedCRUs[currentBuffer][relTF][cru] = true;
      }

      // sending IDCs
      sendOutput(pc, currentOutLane, cru, pc.inputs().get<pmr::vector<float>>(ref));
    }

    LOGP(info, "number of received CRUs for current TF: {}    Needed a total number of processed CRUs of: {}   Current TF: {}", mProcessedCRU[currentBuffer][relTF], mCRUs.size(), tf);

    // check for missing data if specified
    if (mNTFsDataDrop > 0) {
      checkIntervalsForMissingData(pc, currentBuffer, relTF, currentOutLane, tf);
    }

    if (mProcessedCRU[currentBuffer][relTF] == mCRUs.size()) {
      ++mProcessedTFs[currentBuffer];
    }

    if (mProcessedTFs[currentBuffer] == mTimeFrames) {
      finishInterval(pc, currentOutLane, currentBuffer, tf);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final { ec.services().get<ControlService>().readyToQuit(QuitRequest::Me); }

  /// return data description for aggregated IDCs for given lane
  static header::DataDescription getDataDescriptionIDC(const unsigned int lane)
  {
    const std::string name = fmt::format("IDCAGG{}", lane).data();
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

  static constexpr header::DataDescription getDataDescriptionIDCFirstTF() { return header::DataDescription{"IDCFIRSTTF"}; }
  static constexpr header::DataDescription getDataDescriptionIDCOrbitReset() { return header::DataDescription{"IDCORBITRESET"}; }

 private:
  std::vector<uint32_t> mCRUs{};                                                       ///< CRUs to process in this instance
  const unsigned int mTimeFrames{};                                                    ///< number of TFs per aggregation interval
  const int mNTFsBuffer{1};                                                            ///< number of TFs for which the IDCs will be buffered
  const unsigned int mOutLanes{};                                                      ///< number of output lanes
  std::array<unsigned int, 2> mProcessedTFs{{0, 0}};                                   ///< number of processed time frames to keep track of when the writing to CCDB will be done
  std::array<std::vector<unsigned int>, 2> mProcessedCRU{};                            ///< counter of received data from CRUs per TF to merge incoming data from FLPs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::vector<std::unordered_map<unsigned int, bool>>, 2> mProcessedCRUs{}; ///< to keep track of the already processed CRUs ([buffer][relTF][CRU])
  std::array<long, 2> mTFStart{};                                                      ///< storing of first TF for buffer interval
  std::array<long, 2> mTFEnd{};                                                        ///< storing of last TF for buffer interval
  std::array<bool, 2> mSendOutputStartInfo{true, true};                                ///< flag for sending the info for the start of the aggregation interval
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                              ///< info for CCDB request
  std::vector<bool> mSendCCDBOutput{};                                                 ///< flag for sending CCDB output
  unsigned int mCurrentOutLane{0};                                                     ///< index for keeping track of the current output lane
  bool mBuffer{false};                                                                 ///< buffer index
  int mNFactorTFs{0};                                                                  ///< Number of TFs to skip for sending oldest TF
  int mNTFsDataDrop{0};                                                                ///< delay for the check if TFs are missing in TF units
  std::array<int, 2> mStartNTFsDataDrop{0};                                            ///< first relative TF to check
  long mProcessedTotalData{0};                                                         ///< used to check for dropeed TF data
  int mCheckEveryNData{1};                                                             ///< factor after which to check for missing data (in case data missing -> send dummy data)
  std::vector<InputSpec> mFilter{};                                                    ///< filter for looping over input data
  std::vector<header::DataDescription> mDataDescrOut{};

  void sendOutput(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const unsigned int cru, o2::pmr::vector<float> idcs)
  {
    pc.outputs().adoptContainer(Output{gDataOriginTPC, mDataDescrOut[currentOutLane], header::DataHeader::SubSpecificationType{cru}}, std::move(idcs));
  }

  /// returns the output lane to which the data will be send
  unsigned int getOutLane(const uint32_t tf) const { return (tf > mTFEnd[mBuffer]) ? (mCurrentOutLane + 1) % mOutLanes : mCurrentOutLane; }

  /// returns real number of TFs taking buffer size into account
  unsigned int getNRealTFs() const { return mNTFsBuffer * mTimeFrames; }

  void clearBuffer(const bool currentBuffer)
  {
    // resetting received CRUs
    for (auto& crusMap : mProcessedCRUs[currentBuffer]) {
      for (auto& it : crusMap) {
        it.second = false;
      }
    }

    mProcessedTFs[currentBuffer] = 0; // reset processed TFs for next aggregation interval
    std::fill(mProcessedCRU[currentBuffer].begin(), mProcessedCRU[currentBuffer].end(), 0);

    // set integration range for next integration interval
    mTFStart[mBuffer] = mTFEnd[!mBuffer] + 1;
    mTFEnd[mBuffer] = mTFStart[mBuffer] + getNRealTFs() - 1;

    // switch buffer
    mBuffer = !mBuffer;

    // set output lane
    mCurrentOutLane = ++mCurrentOutLane % mOutLanes;
  }

  void checkIntervalsForMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const long relTF, const unsigned int currentOutLane, const uint32_t tf)
  {
    if (!(mProcessedTotalData++ % mCheckEveryNData)) {
      LOGP(info, "Checking for dropped packages...");

      // if last buffer has smaller time range check the whole last buffer
      if ((mTFStart[currentBuffer] > mTFStart[!currentBuffer]) && (relTF > mNTFsDataDrop)) {
        LOGP(warning, "checking last buffer from {} to {}", mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size());
        const unsigned int lastLane = (currentOutLane == 0) ? (mOutLanes - 1) : (currentOutLane - 1);
        checkMissingData(pc, !currentBuffer, mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size(), lastLane);
        LOGP(info, "All empty TFs for TF {} for current buffer filled with dummy and sent. Clearing buffer", tf);
        finishInterval(pc, lastLane, !currentBuffer, tf);
      }

      const int tfEndCheck = std::clamp(static_cast<int>(relTF) - mNTFsDataDrop, 0, static_cast<int>(mProcessedCRU[currentBuffer].size()));
      LOGP(info, "checking current buffer from {} to {}", mStartNTFsDataDrop[currentBuffer], tfEndCheck);
      checkMissingData(pc, currentBuffer, mStartNTFsDataDrop[currentBuffer], tfEndCheck, currentOutLane);
      mStartNTFsDataDrop[currentBuffer] = tfEndCheck;
    }
  }

  void checkMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const int startTF, const int endTF, const unsigned int outLane)
  {
    for (int iTF = startTF; iTF < endTF; ++iTF) {
      if (mProcessedCRU[currentBuffer][iTF] != mCRUs.size()) {
        LOGP(warning, "CRUs for lane {}  rel. TF: {}  curr TF {} are missing! Processed {} CRUs out of {}", outLane, iTF, mTFStart[currentBuffer] + iTF, mProcessedCRU[currentBuffer][iTF], mCRUs.size());
        ++mProcessedTFs[currentBuffer];
        mProcessedCRU[currentBuffer][iTF] = mCRUs.size();

        // find missing CRUs
        for (auto& it : mProcessedCRUs[currentBuffer][iTF]) {
          if (!it.second) {
            it.second = true;
            sendOutput(pc, outLane, it.first, pmr::vector<float>());
          }
        }
      }
    }
  }

  void finishInterval(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const bool buffer, const uint32_t tf)
  {
    if (mNFactorTFs > 0) {
      mNFactorTFs = 0;
      // ToDo: Find better fix
      for (unsigned int ilane = 0; ilane < mOutLanes; ++ilane) {
        auto& deviceProxy = pc.services().get<FairMQDeviceProxy>();
        auto& state = deviceProxy.getOutputChannelState({static_cast<int>(ilane)});
        size_t oldest = std::numeric_limits<size_t>::max() - 1; // just set to really large value
        state.oldestForChannel = {oldest};
      }
    }

    LOGP(info, "All TFs {} for current buffer received. Clearing buffer", tf);
    clearBuffer(buffer);
    mStartNTFsDataDrop[buffer] = 0;
    mSendOutputStartInfo[buffer] = true;
  }
};

DataProcessorSpec getTPCDistributeIDCSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const int firstTF, const bool sendPrecisetimeStamp = false, const int nTFsBuffer = 1)
{
  std::vector<InputSpec> inputSpecs;
  const auto sides = IDCFactorization::getSides(crus);
  for (auto side : sides) {
    const std::string name = (side == Side::A) ? "idcsgroupa" : "idcsgroupc";
    inputSpecs.emplace_back(InputSpec{name.data(), ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPIDCDevice::getDataDescriptionIDCGroup(side)}, Lifetime::Sporadic});
  }

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.reserve(outlanes);
  for (unsigned int lane = 0; lane < outlanes; ++lane) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(lane)}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDCFirstTF(), header::DataHeader::SubSpecificationType{lane}}, Lifetime::Sporadic);
  }

  bool fetchCCDB = false;
  if (sendPrecisetimeStamp && (ilane == 0)) {
    fetchCCDB = true;
    for (unsigned int lane = 0; lane < outlanes; ++lane) {
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDCOrbitReset(), header::DataHeader::SubSpecificationType{lane}}, Lifetime::Sporadic);
    }
  }

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(fetchCCDB,                      // orbitResetTime
                                                                fetchCCDB,                      // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputSpecs);

  const auto id = fmt::format("tpc-distribute-idc-{:02}", ilane);
  DataProcessorSpec spec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCDistributeIDCSpec>(crus, timeframes, nTFsBuffer, outlanes, firstTF, ccdbRequest)},
    Options{{"drop-data-after-nTFs", VariantType::Int, 0, {"Number of TFs after which to drop the data."}},
            {"check-data-every-n", VariantType::Int, 0, {"Number of run function called after which to check for missing data (-1 for no checking, 0 for default checking)."}},
            {"nFactorTFs", VariantType::Int, 1000, {"Number of TFs to skip for sending oldest TF."}}}}; // end DataProcessorSpec
  spec.rank = ilane;
  return spec;
}

} // namespace o2::tpc

#endif
