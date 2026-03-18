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
/// @brief  TPC aggregation of grouped CMVs

#ifndef O2_TPCDISTRIBUTECMVSPEC_H
#define O2_TPCDISTRIBUTECMVSPEC_H

#include <vector>
#include <chrono>
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
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "TPCCalibration/CMVContainer.h"
#include "DataFormatsTPC/CMV.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCDistributeCMVSpec : public o2::framework::Task
{
 public:
  TPCDistributeCMVSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const int nTFsBuffer, const int firstTF, const bool sendCCDB, const bool usePreciseTimestamp, std::shared_ptr<o2::base::GRPGeomRequest> req)
    : mCRUs{crus},
      mTimeFrames{timeframes},
      mNTFsBuffer{nTFsBuffer},
      mProcessedCRU{{std::vector<unsigned int>(timeframes), std::vector<unsigned int>(timeframes)}},
      mTFStart{{firstTF, firstTF + timeframes}},
      mTFEnd{{firstTF + timeframes - 1, mTFStart[1] + timeframes - 1}},
      mCCDBRequest(req),
      mSendCCDB{sendCCDB},
      mUsePreciseTimestamp{usePreciseTimestamp},
      mSendCCDBOutputOrbitReset(1),
      mSendCCDBOutputGRPECS(1),
      mOrbitInfoForwarded{{std::vector<bool>(timeframes, false), std::vector<bool>(timeframes, false)}}
  {
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

    mFilter.emplace_back(InputSpec{"cmvsgroup", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup()}, Lifetime::Sporadic});
    mOrbitFilter.emplace_back(InputSpec{"cmvorbit", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo()}, Lifetime::Sporadic});

    // Pre-allocate CMVPerInterval storage
    mInterval.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
  };

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
    mDumpCMVs = ic.options().get<bool>("dump-cmvs");
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    if (matcher == ConcreteDataMatcher("CTP", "ORBITRESET", 0)) {
      LOGP(info, "Updating ORBITRESET");
      std::fill(mSendCCDBOutputOrbitReset.begin(), mSendCCDBOutputOrbitReset.end(), true);
    } else if (matcher == ConcreteDataMatcher("GLO", "GRPECS", 0)) {
      // check if received object is valid
      if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRun() != 0) {
        LOGP(info, "Updating GRPECS");
        std::fill(mSendCCDBOutputGRPECS.begin(), mSendCCDBOutputGRPECS.end(), true);
      } else {
        LOGP(info, "Detected default GRPECS object");
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
      // Update mTFInfo from GRPGeomHelper whenever orbit-reset or GRPECS objects are fresh
      if (mSendCCDBOutputOrbitReset[0] && mSendCCDBOutputGRPECS[0]) {
        mSendCCDBOutputOrbitReset[0] = false;
        mSendCCDBOutputGRPECS[0] = false;
        mTFInfo = dataformats::Pair<long, int>{o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS(), o2::base::GRPGeomHelper::instance().getNHBFPerTF()};
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
      LOGP(info, "All CRUs for current TF {} already received. Skipping this TF", tf);
      return;
    }

    const unsigned int relTF = (tf - mTFStart[currentBuffer]) / mNTFsBuffer;
    LOGP(info, "Current TF: {}, relative TF: {}, current buffer: {}, mTFStart: {}", tf, relTF, currentBuffer, mTFStart[currentBuffer]);

    if (relTF >= mProcessedCRU[currentBuffer].size()) {
      LOGP(warning, "Skipping tf {}: relative tf {} is larger than size of buffer: {}", tf, relTF, mProcessedCRU[currentBuffer].size());

      // check number of processed CRUs for previous TFs. If CRUs are missing for them, they are probably lost/not received
      mProcessedTotalData = mCheckEveryNData;
      checkIntervalsForMissingData(pc, currentBuffer, relTF, tf);
      return;
    }

    if (mProcessedCRU[currentBuffer][relTF] == mCRUs.size()) {
      return;
    }

    // Record the absolute first TF of this aggregation interval
    if (mInterval.firstTF == 0) {
      mInterval.firstTF = mTFStart[currentBuffer];
    }

    // Set CCDB start timestamp once at the start of each aggregation interval
    if (mTimestampStart == 0) {
      setTimestampCCDB(relTF, pc);
    }

    // Capture orbit/BC info into the interval once per relTF.
    // All CRUs within a TF carry identical timing, so the first one is sufficient.
    if (!mOrbitInfoForwarded[currentBuffer][relTF]) {
      for (auto& ref : InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const unsigned int cru = hdr->subSpecification >> 7;
        if (std::binary_search(mCRUs.begin(), mCRUs.end(), cru)) {
          const auto orbitBC = pc.inputs().get<uint64_t>(ref);
          auto& tfData = mInterval.mCMVPerTF[relTF];
          if (tfData.firstOrbit == 0 && tfData.firstBC == 0) {
            tfData.firstOrbit = static_cast<int64_t>(orbitBC >> 32);
            tfData.firstBC = static_cast<int64_t>(orbitBC & 0xFFFFu);
          }
          mOrbitInfoForwarded[currentBuffer][relTF] = true;
          break; // one per relTF is enough
        }
      }
    }

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = tpcCRUHeader->subSpecification >> 7;

      // check if cru is specified in input cru list
      if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
        LOGP(info, "Received data from CRU: {} which was not specified as input. Skipping", cru);
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

      // accumulate CMVs into the interval
      auto cmvVec = pc.inputs().get<pmr::vector<float>>(ref);
      mInterval.mCMVPerTF[relTF].mDataPerTF[cru].assign(cmvVec.begin(), cmvVec.end());
    }

    LOGP(info, "Number of received CRUs for current TF: {} Needed a total number of processed CRUs of: {} Current TF: {}", mProcessedCRU[currentBuffer][relTF], mCRUs.size(), tf);

    // check for missing data if specified
    if (mNTFsDataDrop > 0) {
      checkIntervalsForMissingData(pc, currentBuffer, relTF, tf);
    }

    if (mProcessedCRU[currentBuffer][relTF] == mCRUs.size()) {
      ++mProcessedTFs[currentBuffer];
    }

    if (mProcessedTFs[currentBuffer] == mTimeFrames) {
      mInterval.lastTF = tf;
      sendOutput(pc.outputs(), tf);
      finishInterval(pc, currentBuffer, tf);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "End of stream, flushing CMV interval ({} TFs)", mInterval.size());
    sendOutput(ec.outputs(), 0);
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionCCDBCMV() { return header::DataDescription{"TPC_CMV"}; }

  /// Return data description for aggregated CMVs for a given lane
  static header::DataDescription getDataDescriptionCMV(const unsigned int lane)
  {
    const std::string name = fmt::format("CMVAGG{}", lane).data();
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

  /// return data description for orbit/BC info for a given output lane
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
  const int mNTFsBuffer{1};                                                            ///< number of TFs for which the CMVs will be buffered
  std::array<unsigned int, 2> mProcessedTFs{{0, 0}};                                   ///< number of processed time frames to keep track of when the writing to CCDB will be done
  std::array<std::vector<unsigned int>, 2> mProcessedCRU{};                            ///< counter of received data from CRUs per TF to merge incoming data from FLPs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::vector<std::unordered_map<unsigned int, bool>>, 2> mProcessedCRUs{}; ///< to keep track of the already processed CRUs ([buffer][relTF][CRU])
  std::array<long, 2> mTFStart{};                                                      ///< storing of first TF for buffer interval
  std::array<long, 2> mTFEnd{};                                                        ///< storing of last TF for buffer interval
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                              ///< info for CCDB request
  std::vector<bool> mSendCCDBOutputOrbitReset{};                                       ///< flag for received orbit reset time from CCDB
  std::vector<bool> mSendCCDBOutputGRPECS{};                                           ///< flag for received orbit GRPECS from CCDB
  bool mBuffer{false};                                                                 ///< buffer index
  const bool mSendCCDB{false};                                                         ///< send output to CCDB populator
  const bool mUsePreciseTimestamp{false};                                              ///< use precise timestamp from orbit-reset info
  bool mDumpCMVs{false};                                                               ///< write a local ROOT debug file
  long mTimestampStart{0};                                                             ///< CCDB validity start timestamp
  dataformats::Pair<long, int> mTFInfo{};                                              ///< orbit-reset time and NHBFPerTF for precise timestamp
  CMVPerInterval mInterval{};                                                          ///< accumulated CMV data for the current aggregation interval
  int mNFactorTFs{0};                                                                  ///< Number of TFs to skip for sending oldest TF
  int mNTFsDataDrop{0};                                                                ///< delay for the check if TFs are missing in TF units
  std::array<int, 2> mStartNTFsDataDrop{0};                                            ///< first relative TF to check
  long mProcessedTotalData{0};                                                         ///< used to check for dropeed TF data
  int mCheckEveryNData{1};                                                             ///< factor after which to check for missing data (in case data missing -> send dummy data)
  std::vector<InputSpec> mFilter{};                                                    ///< filter for looping over input data
  std::vector<InputSpec> mOrbitFilter{};                                               ///< filter for CMVORBITINFO from FLP
  std::array<std::vector<bool>, 2> mOrbitInfoForwarded{};                              ///< tracks whether orbit/BC has been captured per (buffer, relTF)

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
    std::fill(mOrbitInfoForwarded[currentBuffer].begin(), mOrbitInfoForwarded[currentBuffer].end(), false);

    // set integration range for next integration interval
    mTFStart[mBuffer] = mTFEnd[!mBuffer] + 1;
    mTFEnd[mBuffer] = mTFStart[mBuffer] + getNRealTFs() - 1;

    // switch buffer
    mBuffer = !mBuffer;
  }

  void checkIntervalsForMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const long relTF, const uint32_t tf)
  {
    if (!(mProcessedTotalData++ % mCheckEveryNData)) {
      LOGP(info, "Checking for dropped packages...");

      // if last buffer has smaller time range check the whole last buffer
      if ((mTFStart[currentBuffer] > mTFStart[!currentBuffer]) && (relTF > mNTFsDataDrop)) {
        LOGP(warning, "Checking last buffer from {} to {}", mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size());
        checkMissingData(pc, !currentBuffer, mStartNTFsDataDrop[!currentBuffer], mProcessedCRU[!currentBuffer].size());
        LOGP(info, "All empty TFs for TF {} for current buffer filled with dummy and sent. Clearing buffer", tf);
        mInterval.lastTF = tf;
        sendOutput(pc.outputs(), tf);
        finishInterval(pc, !currentBuffer, tf);
      }

      const int tfEndCheck = std::clamp(static_cast<int>(relTF) - mNTFsDataDrop, 0, static_cast<int>(mProcessedCRU[currentBuffer].size()));
      LOGP(info, "Checking current buffer from {} to {}", mStartNTFsDataDrop[currentBuffer], tfEndCheck);
      checkMissingData(pc, currentBuffer, mStartNTFsDataDrop[currentBuffer], tfEndCheck);
      mStartNTFsDataDrop[currentBuffer] = tfEndCheck;
    }
  }

  void checkMissingData(o2::framework::ProcessingContext& pc, const bool currentBuffer, const int startTF, const int endTF)
  {
    for (int iTF = startTF; iTF < endTF; ++iTF) {
      if (mProcessedCRU[currentBuffer][iTF] != mCRUs.size()) {
        LOGP(warning, "CRUs for rel. TF: {}  curr TF {} are missing! Processed {} CRUs out of {}", iTF, mTFStart[currentBuffer] + iTF, mProcessedCRU[currentBuffer][iTF], mCRUs.size());
        ++mProcessedTFs[currentBuffer];
        mProcessedCRU[currentBuffer][iTF] = mCRUs.size();

        // find missing CRUs and leave their interval slots empty (zero-filled)
        for (auto& it : mProcessedCRUs[currentBuffer][iTF]) {
          if (!it.second) {
            it.second = true;
          }
        }

        // leave orbit/BC as zero placeholder for missing TFs
        mOrbitInfoForwarded[currentBuffer][iTF] = true;
      }
    }
  }

  void finishInterval(o2::framework::ProcessingContext& pc, const bool buffer, const uint32_t tf)
  {
    if (mNFactorTFs > 0) {
      mNFactorTFs = 0;
      // ToDo: Find better fix
      auto& deviceProxy = pc.services().get<FairMQDeviceProxy>();
      auto& state = deviceProxy.getOutputChannelState({0});
      size_t oldest = std::numeric_limits<size_t>::max() - 1; // just set to really large value
      state.oldestForChannel = {oldest};
    }

    LOGP(info, "All TFs {} for current buffer received. Clearing buffer", tf);
    clearBuffer(buffer);
    mStartNTFsDataDrop[buffer] = 0;

    // Reset per-interval state for the next aggregation interval
    mInterval.clear();
    mInterval.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
    mTimestampStart = 0;
    LOGP(info, "Everything cleared. Waiting for new data to arrive.");
  }

  void setTimestampCCDB(const long relTF, o2::framework::ProcessingContext& pc)
  {
    if (mUsePreciseTimestamp && !mTFInfo.second) {
      return;
    }
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    const auto nOrbitsOffset = (relTF * mNTFsBuffer + (mNTFsBuffer - 1)) * mTFInfo.second;
    mTimestampStart = mUsePreciseTimestamp
                        ? (mTFInfo.first + (tinfo.firstTForbit - nOrbitsOffset) * o2::constants::lhc::LHCOrbitMUS * 0.001)
                        : tinfo.creation;
    LOGP(info, "Setting timestamp reset reference to: {}, at tfCounter: {}, firstTForbit: {}, NHBFPerTF: {}, relTF: {}, nOrbitsOffset: {}",
         mTFInfo.first, tinfo.tfCounter, tinfo.firstTForbit, mTFInfo.second, relTF, nOrbitsOffset);
  }

  void sendOutput(DataAllocator& output, const uint32_t tf)
  {
    using timer = std::chrono::high_resolution_clock;

    if (mInterval.empty()) {
      LOGP(warning, "CMV interval is empty at sendOutput, skipping");
      return;
    }

    // Check if any CRU actually wrote data into this interval
    const bool hasData = std::any_of(mInterval.mCMVPerTF.begin(), mInterval.mCMVPerTF.end(),
                                     [](const CMVPerTF& tfd) {
                                       return std::any_of(tfd.mDataPerTF.begin(), tfd.mDataPerTF.end(),
                                                          [](const std::vector<float>& v) { return !v.empty(); });
                                     });
    if (!hasData) {
      LOGP(warning, "CMV interval has no data at sendOutput, skipping");
      return;
    }

    LOGP(info, "{}", mInterval.summary());
    auto start = timer::now();
    auto tree = mInterval.toTTree();

    // Write local ROOT file for debugging
    if (mDumpCMVs) {
      const std::string fname = fmt::format("CMV_timestamp{}.root", mTimestampStart);
      try {
        mInterval.writeToFile(fname, tree);
        LOGP(info, "CMV debug file written to {}", fname);
      } catch (const std::exception& e) {
        LOGP(error, "Failed to write CMV debug file: {}", e.what());
      }
    }

    if (!mSendCCDB) {
      LOGP(warning, "CCDB output disabled, skipping upload!");
      return;
    }

    const int nHBFPerTF = o2::base::GRPGeomHelper::instance().getNHBFPerTF();
    const long timeStampEnd = mTimestampStart + static_cast<long>(mTimeFrames * mNTFsBuffer * nHBFPerTF * o2::constants::lhc::LHCOrbitMUS * 1e-3);

    if (timeStampEnd <= mTimestampStart) {
      LOGP(warning, "Invalid CCDB timestamp range start:{} end:{}, skipping upload!",
           mTimestampStart, timeStampEnd);
      return;
    }

    LOGP(info, "CCDB timestamp range start:{} end:{}", mTimestampStart, timeStampEnd);

    o2::ccdb::CcdbObjectInfo ccdbInfoCMV(
      "TPC/Calib/CMV",
      "TTree",
      "CMV.root",
      {},
      mTimestampStart,
      timeStampEnd);

    auto image = o2::ccdb::CcdbApi::createObjectImage((tree.get()), &ccdbInfoCMV);
    LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {}",
         ccdbInfoCMV.getPath(), ccdbInfoCMV.getFileName(), image->size(),
         ccdbInfoCMV.getStartValidityTimestamp(), ccdbInfoCMV.getEndValidityTimestamp());

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBCMV(), 0}, *image);
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBCMV(), 0}, ccdbInfoCMV);

    auto stop = timer::now();
    std::chrono::duration<float> elapsed = stop - start;
    LOGP(info, "CMV CCDB serialisation time: {:.3f} s", elapsed.count());
  }
};

DataProcessorSpec getTPCDistributeCMVSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const int firstTF, const bool sendCCDB = false, const bool usePreciseTimestamp = false, const int nTFsBuffer = 1)
{
  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"cmvsgroup", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup()}, Lifetime::Sporadic});
  inputSpecs.emplace_back(InputSpec{"cmvorbit", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo()}, Lifetime::Sporadic});

  std::vector<OutputSpec> outputSpecs;
  if (sendCCDB) {
    outputSpecs.emplace_back(
      ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload,
                              TPCDistributeCMVSpec::getDataDescriptionCCDBCMV()},
      Lifetime::Sporadic);
    outputSpecs.emplace_back(
      ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper,
                              TPCDistributeCMVSpec::getDataDescriptionCCDBCMV()},
      Lifetime::Sporadic);
  }

  const bool fetchCCDB = usePreciseTimestamp;
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(fetchCCDB,                      // orbitResetTime
                                                                fetchCCDB,                      // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputSpecs);

  const std::string type = "cmv";
  const auto id = fmt::format("tpc-distribute-{}-{:02}", type, ilane);
  DataProcessorSpec spec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCDistributeCMVSpec>(crus, timeframes, nTFsBuffer, firstTF, sendCCDB, usePreciseTimestamp, ccdbRequest)},
    Options{{"drop-data-after-nTFs", VariantType::Int, 0, {"Number of TFs after which to drop the data."}},
            {"check-data-every-n", VariantType::Int, 0, {"Number of run function called after which to check for missing data (-1 for no checking, 0 for default checking)."}},
            {"nFactorTFs", VariantType::Int, 1000, {"Number of TFs to skip for sending oldest TF."}},
            {"dump-cmvs", VariantType::Bool, false, {"Dump CMVs to a local ROOT file for debugging"}}}}; // end DataProcessorSpec
  spec.rank = ilane;
  return spec;
}

} // namespace o2::tpc

#endif