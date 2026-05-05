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

/// @file   TPCDistributeCMVSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  TPC distribution of grouped CMVs towards the CMV aggregation workflow

#ifndef O2_TPCDISTRIBUTECMVSPEC_H
#define O2_TPCDISTRIBUTECMVSPEC_H

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataTakingContext.h"
#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/TPCFLPCMVSpec.h"
#include "MemoryResources/MemoryResources.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonDataFormat/Pair.h"

namespace o2::tpc
{

class TPCDistributeCMVSpec : public o2::framework::Task
{
 public:
  TPCDistributeCMVSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const int nTFsBuffer, const unsigned int outlanes, const int firstTF, std::shared_ptr<o2::base::GRPGeomRequest> req)
    : mCRUs{crus},
      mTimeFrames{timeframes},
      mNTFsBuffer{nTFsBuffer},
      mOutLanes{outlanes},
      mProcessedCRU{{std::vector<unsigned int>(timeframes), std::vector<unsigned int>(timeframes)}},
      mTFStart{{firstTF, firstTF + static_cast<long>(timeframes) * nTFsBuffer}},
      mTFEnd{{firstTF + static_cast<long>(timeframes) * nTFsBuffer - 1, firstTF + 2LL * timeframes * nTFsBuffer - 1}},
      mCCDBRequest(req),
      mSendCCDBOutputOrbitReset(outlanes),
      mSendCCDBOutputGRPECS(outlanes),
      mOrbitInfoForwarded{{std::vector<bool>(timeframes, false), std::vector<bool>(timeframes, false)}}
  {
    mDataDescrOut.reserve(mOutLanes);
    mOrbitDescrOut.reserve(mOutLanes);
    for (unsigned int i = 0; i < mOutLanes; ++i) {
      mDataDescrOut.emplace_back(getDataDescriptionCMV(i));
      mOrbitDescrOut.emplace_back(getDataDescriptionCMVOrbitInfo(i));
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

    mFilter.emplace_back(o2::framework::InputSpec{"cmvsgroup", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup()}, o2::framework::Lifetime::Sporadic});
    mOrbitFilter.emplace_back(o2::framework::InputSpec{"cmvorbit", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo()}, o2::framework::Lifetime::Sporadic});
  }

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNFactorTFs = ic.options().get<int>("nFactorTFs");
    mNTFsDataDrop = ic.options().get<int>("drop-data-after-nTFs");
    mCheckEveryNData = ic.options().get<int>("check-data-every-n");
    if (mCheckEveryNData == 0) {
      mCheckEveryNData = mTimeFrames / 2;
      if (mCheckEveryNData == 0) {
        mCheckEveryNData = 1;
      }
      mNTFsDataDrop = mCheckEveryNData;
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    if (matcher == o2::framework::ConcreteDataMatcher("CTP", "ORBITRESET", 0)) {
      LOGP(debug, "Updating ORBITRESET");
      std::fill(mSendCCDBOutputOrbitReset.begin(), mSendCCDBOutputOrbitReset.end(), true);
    } else if (matcher == o2::framework::ConcreteDataMatcher("GLO", "GRPECS", 0)) {
      // check if received object is valid
      if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRun() != 0) {
        LOGP(debug, "Updating GRPECS");
        std::fill(mSendCCDBOutputGRPECS.begin(), mSendCCDBOutputGRPECS.end(), true);
      } else {
        LOGP(debug, "Detected default GRPECS object");
      }
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // capture orbit-reset info once for precise CCDB timestamp calculation
    if (mCCDBRequest->askTime) {
      const bool grpecsValid = pc.inputs().isValid("grpecs");
      const bool orbitResetValid = pc.inputs().isValid("orbitReset");
      if (grpecsValid) {
        pc.inputs().get<o2::parameters::GRPECSObject*>("grpecs");
      }
      if (orbitResetValid) {
        pc.inputs().get<std::vector<Long64_t>*>("orbitReset");
      }
      if (pc.inputs().countValidInputs() == (grpecsValid + orbitResetValid)) {
        return;
      }
    }

    const auto tf = processing_helpers::getCurrentTF(pc);
    if (tf == std::numeric_limits<uint32_t>::max()) {
      forwardEOSData(pc);
      return;
    }

    // automatically detect firstTF in case firstTF was not specified
    if (mTFStart.front() <= -1) {
      const auto firstTFDetected = tf;
      const long offsetTF = std::abs(mTFStart.front() + 1);
      const auto nTotTFs = getNRealTFs();
      // tf is the batch TF counter (= last real TF in the first batch), subtract (mNTFsBuffer - 1) to recover the actual first real TF of the interval
      const long firstRealTF = static_cast<long>(firstTFDetected) - (mNTFsBuffer - 1) + offsetTF;
      mTFStart = {firstRealTF, firstRealTF + nTotTFs};
      mTFEnd = {mTFStart[1] - 1, mTFStart[1] - 1 + nTotTFs};
      LOGP(detail, "Setting {} as first TF", mTFStart[0]);
      LOGP(detail, "Using offset of {} TFs for setting the first TF", offsetTF);
    }

    // check which buffer to use for current incoming data
    const bool currentBuffer = (tf > mTFEnd[mBuffer]) ? !mBuffer : mBuffer;
    if (mTFStart[currentBuffer] > tf) {
      LOGP(detail, "All CRUs for current TF {} already received. Skipping this TF", tf);
      return;
    }

    const unsigned int currentOutLane = getOutLane(tf);
    const unsigned int relTF = (tf - mTFStart[currentBuffer]) / mNTFsBuffer;
    LOGP(debug, "Current TF: {}, relative TF: {}, current buffer: {}, current output lane: {}, mTFStart: {}", tf, relTF, currentBuffer, currentOutLane, mTFStart[currentBuffer]);

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

    if (mSendOutputStartInfo[currentBuffer]) {
      mSendOutputStartInfo[currentBuffer] = false;
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVFirstTF(), header::DataHeader::SubSpecificationType{currentOutLane}}, mTFStart[currentBuffer]);
    }

    if (mSendCCDBOutputOrbitReset[currentOutLane] && mSendCCDBOutputGRPECS[currentOutLane]) {
      mSendCCDBOutputOrbitReset[currentOutLane] = false;
      mSendCCDBOutputGRPECS[currentOutLane] = false;
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVOrbitReset(), header::DataHeader::SubSpecificationType{currentOutLane}}, dataformats::Pair<long, int>{o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS(), o2::base::GRPGeomHelper::instance().getNHBFPerTF()});
    }

    forwardOrbitInfo(pc, currentBuffer, relTF, currentOutLane);

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = tpcCRUHeader->subSpecification >> 7;

      // check if cru is specified in input cru list
      if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
        LOGP(debug, "Received data from CRU: {} which was not specified as input. Skipping", cru);
        continue;
      }

      if (mProcessedCRUs[currentBuffer][relTF][cru]) {
        continue;
      }
      // count total number of processed CRUs for given TF
      ++mProcessedCRU[currentBuffer][relTF];
      // to keep track of processed CRUs
      mProcessedCRUs[currentBuffer][relTF][cru] = true;

      sendOutput(pc, currentOutLane, cru, pc.inputs().get<o2::pmr::vector<uint16_t>>(ref));
    }

    LOGP(detail, "Number of received CRUs for current TF: {} Needed a total number of processed CRUs of: {} Current TF: {}", mProcessedCRU[currentBuffer][relTF], mCRUs.size(), tf);

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

  void endOfStream(o2::framework::EndOfStreamContext& ec) final { ec.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me); }

  /// Return data description for aggregated CMVs for a given lane
  static header::DataDescription getDataDescriptionCMV(const unsigned int lane)
  {
    const std::string name = fmt::format("CMVAGG{}", lane);
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

  /// Return data description for orbit/BC info for a given output lane
  static header::DataDescription getDataDescriptionCMVOrbitInfo(const unsigned int lane)
  {
    const std::string name = fmt::format("CMVORB{}", lane);
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

  static constexpr header::DataDescription getDataDescriptionCMVFirstTF() { return header::DataDescription{"CMVFIRSTTF"}; }
  static constexpr header::DataDescription getDataDescriptionCMVOrbitReset() { return header::DataDescription{"CMVORBITRESET"}; }

 private:
  std::vector<uint32_t> mCRUs{};                                                       ///< CRUs to process in this instance
  const unsigned int mTimeFrames{};                                                    ///< number of TFs per aggregation interval
  const int mNTFsBuffer{1};                                                            ///< number of TFs for which the CMVs will be buffered (must match TPCFLPCMVSpec)
  const unsigned int mOutLanes{};                                                      ///< number of parallel aggregate pipelines this distributor feeds
  std::array<unsigned int, 2> mProcessedTFs{{0, 0}};                                   ///< number of processed timeframes per buffer; triggers sendOutput when it reaches mTimeFrames
  std::array<std::vector<unsigned int>, 2> mProcessedCRU{};                            ///< counter of received CRUs per (buffer, relTF); used to detect when a relTF is complete
  std::array<std::vector<std::unordered_map<unsigned int, bool>>, 2> mProcessedCRUs{}; ///< per-CRU received flag ([buffer][relTF][CRU]); prevents double-counting when a CRU re-sends
  std::array<long, 2> mTFStart{};                                                      ///< absolute TF counter of the first TF in each buffer interval
  std::array<long, 2> mTFEnd{};                                                        ///< absolute TF counter of the last TF in each buffer interval
  std::array<bool, 2> mSendOutputStartInfo{true, true};                                ///< flag to send CMVFIRSTTF message once at the start of each buffer interval
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                              ///< info for CCDB request (orbit-reset and GRPECS, only on lane 0 when sendPreciseTimestamp=true)
  std::vector<bool> mSendCCDBOutputOrbitReset{};                                       ///< per-output-lane flag: true when a fresh orbit-reset object has been received from CCDB
  std::vector<bool> mSendCCDBOutputGRPECS{};                                           ///< per-output-lane flag: true when a fresh GRPECS object has been received from CCDB
  unsigned int mCurrentOutLane{0};                                                     ///< output lane currently being filled
  bool mBuffer{false};                                                                 ///< double-buffer index (false = buffer 0, true = buffer 1)
  int mNFactorTFs{0};                                                                  ///< number of TFs to skip when setting oldestForChannel; resets to 0 after first interval
  int mNTFsDataDrop{0};                                                                ///< delay (in relTF units) before declaring a relTF's missing CRUs as lost
  std::array<int, 2> mStartNTFsDataDrop{0};                                            ///< first relative TF index to check for missing data in each buffer
  long mProcessedTotalData{0};                                                         ///< call counter used to throttle checkIntervalsForMissingData checks
  int mCheckEveryNData{1};                                                             ///< check for missing data every N run() calls (0 → default = mTimeFrames/2)
  std::vector<o2::framework::InputSpec> mFilter{};                                     ///< filter for looping over CMVGROUP input data from FLPs
  std::vector<o2::framework::InputSpec> mOrbitFilter{};                                ///< filter for CMVORBITINFO input from FLPs
  std::vector<header::DataDescription> mDataDescrOut{};                                ///< per-output-lane CMV data descriptions (CMVAGG0, CMVAGG1, …)
  std::vector<header::DataDescription> mOrbitDescrOut{};                               ///< per-output-lane orbit-info data descriptions (CMVORB0, CMVORB1, …)
  std::array<std::vector<bool>, 2> mOrbitInfoForwarded{};                              ///< tracks whether orbit/BC has been forwarded to the aggregate lane per (buffer, relTF)

  /// Returns the output aggregate lane for a given TF counter (advances when the current buffer interval has ended)
  unsigned int getOutLane(const uint32_t tf) const { return (tf > mTFEnd[mBuffer]) ? (mCurrentOutLane + 1) % mOutLanes : mCurrentOutLane; }
  /// Returns the total number of real TFs per buffer interval (= mNTFsBuffer * mTimeFrames)
  unsigned int getNRealTFs() const { return mNTFsBuffer * mTimeFrames; }

  void sendOutput(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const unsigned int cru, o2::pmr::vector<uint16_t> cmvs)
  {
    pc.outputs().adoptContainer(o2::framework::Output{o2::header::gDataOriginTPC, mDataDescrOut[currentOutLane], header::DataHeader::SubSpecificationType{cru}}, std::move(cmvs));
  }

  void sendOrbitInfo(o2::framework::ProcessingContext& pc, const unsigned int outLane, const uint64_t orbitInfo)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, mOrbitDescrOut[outLane], header::DataHeader::SubSpecificationType{outLane}}, orbitInfo);
  }

  void forwardOrbitInfo(o2::framework::ProcessingContext& pc, const bool currentBuffer, const unsigned int relTF, const unsigned int currentOutLane)
  {
    if (mOrbitInfoForwarded[currentBuffer][relTF]) {
      return;
    }

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mOrbitFilter)) {
      auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = hdr->subSpecification >> 7;
      if (!std::binary_search(mCRUs.begin(), mCRUs.end(), cru)) {
        continue;
      }

      sendOrbitInfo(pc, currentOutLane, pc.inputs().get<uint64_t>(ref));
      mOrbitInfoForwarded[currentBuffer][relTF] = true;
      break;
    }
  }

  void forwardEOSData(o2::framework::ProcessingContext& pc)
  {
    const unsigned int currentOutLane = mCurrentOutLane;

    if (mSendOutputStartInfo[mBuffer] && (mTFStart[mBuffer] >= 0)) {
      mSendOutputStartInfo[mBuffer] = false;
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVFirstTF(), header::DataHeader::SubSpecificationType{currentOutLane}}, mTFStart[mBuffer]);
    }

    if (mSendCCDBOutputOrbitReset[currentOutLane] && mSendCCDBOutputGRPECS[currentOutLane]) {
      mSendCCDBOutputOrbitReset[currentOutLane] = false;
      mSendCCDBOutputGRPECS[currentOutLane] = false;
      pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVOrbitReset(), header::DataHeader::SubSpecificationType{currentOutLane}}, dataformats::Pair<long, int>{o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS(), o2::base::GRPGeomHelper::instance().getNHBFPerTF()});
    }

    if (!mOrbitInfoForwarded[mBuffer].empty()) {
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const unsigned int cru = hdr->subSpecification >> 7;
        if (!std::binary_search(mCRUs.begin(), mCRUs.end(), cru)) {
          continue;
        }
        sendOrbitInfo(pc, currentOutLane, pc.inputs().get<uint64_t>(ref));
        break;
      }
    }

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = hdr->subSpecification >> 7;
      if (!std::binary_search(mCRUs.begin(), mCRUs.end(), cru)) {
        continue;
      }
      sendOutput(pc, currentOutLane, cru, pc.inputs().get<o2::pmr::vector<uint16_t>>(ref));
    }
  }

  void clearBuffer(const bool currentBuffer)
  {
    // reset per-CRU received flags so the next interval can accept data from all CRUs again
    for (auto& crusMap : mProcessedCRUs[currentBuffer]) {
      for (auto& it : crusMap) {
        it.second = false;
      }
    }

    mProcessedTFs[currentBuffer] = 0;
    std::fill(mProcessedCRU[currentBuffer].begin(), mProcessedCRU[currentBuffer].end(), 0);
    std::fill(mOrbitInfoForwarded[currentBuffer].begin(), mOrbitInfoForwarded[currentBuffer].end(), false);

    mTFStart[mBuffer] = mTFEnd[!mBuffer] + 1;
    mTFEnd[mBuffer] = mTFStart[mBuffer] + getNRealTFs() - 1;

    // switch buffer and advance output lane
    mBuffer = !mBuffer;
    mCurrentOutLane = ++mCurrentOutLane % mOutLanes;
  }

  void checkIntervalsForMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const long relTF, const unsigned int currentOutLane, const uint32_t tf)
  {
    if (!(mProcessedTotalData++ % mCheckEveryNData)) {
      LOGP(detail, "Checking for dropped packages...");

      // if the last buffer has a smaller time range than expected, flush its remaining uncompleted TFs
      if ((mTFStart[currentBuffer] > mTFStart[!currentBuffer]) && (relTF > mNTFsDataDrop)) {
        LOGP(warning, "Checking last buffer from {} to {}", mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size());
        const unsigned int lastLane = (currentOutLane == 0) ? (mOutLanes - 1) : (currentOutLane - 1);
        checkMissingData(pc, !currentBuffer, mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size(), lastLane);
        LOGP(detail, "All empty TFs for TF {} for current buffer filled with dummy and sent. Clearing buffer", tf);
        finishInterval(pc, lastLane, !currentBuffer, tf);
      }

      const int tfEndCheck = std::clamp(static_cast<int>(relTF) - mNTFsDataDrop, 0, static_cast<int>(mProcessedCRU[currentBuffer].size()));
      LOGP(detail, "Checking current buffer from {} to {}", mStartNTFsDataDrop[currentBuffer], tfEndCheck);
      checkMissingData(pc, currentBuffer, mStartNTFsDataDrop[currentBuffer], tfEndCheck, currentOutLane);
      mStartNTFsDataDrop[currentBuffer] = tfEndCheck;
    }
  }

  void checkMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const int startTF, const int endTF, const unsigned int outLane)
  {
    for (int iTF = startTF; iTF < endTF; ++iTF) {
      if (mProcessedCRU[currentBuffer][iTF] != mCRUs.size()) {
        LOGP(warning, "CRUs for lane {} rel. TF: {} curr TF {} are missing! Processed {} CRUs out of {}", outLane, iTF, mTFStart[currentBuffer] + static_cast<long>(iTF) * mNTFsBuffer, mProcessedCRU[currentBuffer][iTF], mCRUs.size());
        ++mProcessedTFs[currentBuffer];
        mProcessedCRU[currentBuffer][iTF] = mCRUs.size();

        // send empty payloads for missing CRUs so the aggregate lane sees a complete set
        for (auto& it : mProcessedCRUs[currentBuffer][iTF]) {
          if (!it.second) {
            it.second = true;
            sendOutput(pc, outLane, it.first, o2::pmr::vector<uint16_t>());
          }
        }

        // send zero orbit placeholder for missing TF so the aggregate lane can still reconstruct timing
        if (!mOrbitInfoForwarded[currentBuffer][iTF]) {
          sendOrbitInfo(pc, outLane, 0);
          mOrbitInfoForwarded[currentBuffer][iTF] = true;
        }
      }
    }
  }

  void finishInterval(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const bool buffer, const uint32_t tf)
  {
    if (mNFactorTFs > 0) {
      mNFactorTFs = 0;
      // ToDo: Find better fix. Set oldestForChannel to a very large value so the DPL dispatcher does not block waiting for older TF data that will never arrive
      for (unsigned int ilane = 0; ilane < mOutLanes; ++ilane) {
        auto& deviceProxy = pc.services().get<o2::framework::FairMQDeviceProxy>();
        auto& state = deviceProxy.getOutputChannelState({static_cast<int>(ilane)});
        size_t oldest = std::numeric_limits<size_t>::max() - 1;
        state.oldestForChannel = {oldest};
      }
    }

    LOGP(detail, "All TFs {} for current buffer received. Clearing buffer", tf);
    clearBuffer(buffer);
    mStartNTFsDataDrop[buffer] = 0;
    mSendOutputStartInfo[buffer] = true;
  }
};

o2::framework::DataProcessorSpec getTPCDistributeCMVSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const int firstTF, const bool sendPrecisetimeStamp = false, const int nTFsBuffer = 1)
{
  std::vector<o2::framework::InputSpec> inputSpecs;
  inputSpecs.emplace_back(o2::framework::InputSpec{"cmvsgroup", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup()}, o2::framework::Lifetime::Sporadic});
  inputSpecs.emplace_back(o2::framework::InputSpec{"cmvorbit", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo()}, o2::framework::Lifetime::Sporadic});

  std::vector<o2::framework::OutputSpec> outputSpecs;
  outputSpecs.reserve(3 * outlanes);
  for (unsigned int lane = 0; lane < outlanes; ++lane) {
    outputSpecs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(lane)}, o2::framework::Lifetime::Sporadic);
    outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitInfo(lane), header::DataHeader::SubSpecificationType{lane}}, o2::framework::Lifetime::Sporadic);
    outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVFirstTF(), header::DataHeader::SubSpecificationType{lane}}, o2::framework::Lifetime::Sporadic);
  }

  // Only lane 0 fetches CCDB orbit-reset/GRPECS objects and broadcasts them to all aggregate lanes, the other distribute lanes do not need them, avoiding redundant CCDB requests
  bool fetchCCDB = false;
  if (sendPrecisetimeStamp && (ilane == 0)) {
    fetchCCDB = true;
    for (unsigned int lane = 0; lane < outlanes; ++lane) {
      outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitReset(), header::DataHeader::SubSpecificationType{lane}}, o2::framework::Lifetime::Sporadic);
    }
  }

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(fetchCCDB,                      // orbitResetTime
                                                                fetchCCDB,                      // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputSpecs);

  const auto id = fmt::format("tpc-distribute-cmv-{:02}", ilane);
  o2::framework::DataProcessorSpec spec{
    id.data(),
    inputSpecs,
    outputSpecs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<TPCDistributeCMVSpec>(crus, timeframes, nTFsBuffer, outlanes, firstTF, ccdbRequest)},
    o2::framework::Options{{"drop-data-after-nTFs", o2::framework::VariantType::Int, 0, {"Number of TFs after which to drop the data."}},
                           {"check-data-every-n", o2::framework::VariantType::Int, 0, {"Number of run function called after which to check for missing data (-1 for no checking, 0 for default checking)."}},
                           {"nFactorTFs", o2::framework::VariantType::Int, 1000, {"Number of TFs to skip for sending oldest TF."}}}};
  spec.rank = ilane;
  return spec;
}

} // namespace o2::tpc

#endif
