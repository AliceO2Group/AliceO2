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
#include "TParameter.h"
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
#include "TMemFile.h"
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

    // pre-allocate the accumulator TTree for the current aggregation interval
    initIntervalTree();
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
    mUseCompressionVarint = ic.options().get<bool>("use-compression-varint");
    mUseSparse = ic.options().get<bool>("use-sparse");
    mUseCompressionHuffman = ic.options().get<bool>("use-compression-huffman");
    mRoundIntegersThreshold = static_cast<uint16_t>(ic.options().get<int>("cmv-round-integers-threshold"));
    mZeroThreshold = ic.options().get<float>("cmv-zero-threshold");
    mDynamicPrecisionMean = ic.options().get<float>("cmv-dynamic-precision-mean");
    mDynamicPrecisionSigma = ic.options().get<float>("cmv-dynamic-precision-sigma");
    LOGP(info, "CMV compression settings: use-compression-varint={}, use-sparse={}, use-compression-huffman={}, cmv-round-integers-threshold={}, cmv-zero-threshold={}, cmv-dynamic-precision-mean={}, cmv-dynamic-precision-sigma={}",
         mUseCompressionVarint, mUseSparse, mUseCompressionHuffman, mRoundIntegersThreshold, mZeroThreshold, mDynamicPrecisionMean, mDynamicPrecisionSigma);
    // re-initialise the interval tree now that compression options are known (constructor used the defaults)
    initIntervalTree();
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
      // update mTFInfo from GRPGeomHelper whenever orbit-reset or GRPECS objects are fresh
      if (mSendCCDBOutputOrbitReset[0] && mSendCCDBOutputGRPECS[0]) {
        mSendCCDBOutputOrbitReset[0] = false;
        mSendCCDBOutputGRPECS[0] = false;
        mTFInfo = dataformats::Pair<long, int>{o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS(), o2::base::GRPGeomHelper::instance().getNHBFPerTF()};
      }
    }

    const auto tf = processing_helpers::getCurrentTF(pc);
    mLastSeenTF = tf; // track for endOfStream flush

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

    // record the absolute first TF of this aggregation interval
    if (mIntervalTFCount == 0) {
      mIntervalFirstTF = tf;
    }

    // set CCDB start timestamp once at the start of each aggregation interval
    if (mTimestampStart == 0) {
      setTimestampCCDB(relTF, pc);
    }

    // capture orbit/BC info into the interval once per relTF.
    // all CRUs within a TF carry identical timing, so the first one is sufficient.
    if (!mOrbitInfoForwarded[currentBuffer][relTF]) {
      for (auto& ref : InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const unsigned int cru = hdr->subSpecification >> 7;
        if (std::binary_search(mCRUs.begin(), mCRUs.end(), cru)) {
          const auto orbitBC = pc.inputs().get<uint64_t>(ref);
          if (mCurrentTF.firstOrbit == 0 && mCurrentTF.firstBC == 0) {
            mCurrentTF.firstOrbit = static_cast<uint32_t>(orbitBC >> 32);
            mCurrentTF.firstBC = static_cast<uint16_t>(orbitBC & 0xFFFFu);
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

      // accumulate raw 16-bit CMVs into the flat array for the current TF
      auto cmvVec = pc.inputs().get<pmr::vector<uint16_t>>(ref);
      const uint32_t nTimeBins = std::min(static_cast<uint32_t>(cmvVec.size()), cmv::NTimeBinsPerTF);
      for (uint32_t tb = 0; tb < nTimeBins; ++tb) {
        mCurrentTF.mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] = cmvVec[tb];
      }
    }

    LOGP(info, "Number of received CRUs for current TF: {} Needed a total number of processed CRUs of: {} Current TF: {}", mProcessedCRU[currentBuffer][relTF], mCRUs.size(), tf);

    // check for missing data if specified
    if (mNTFsDataDrop > 0) {
      checkIntervalsForMissingData(pc, currentBuffer, relTF, tf);
    }

    if (mProcessedCRU[currentBuffer][relTF] == mCRUs.size()) {
      ++mProcessedTFs[currentBuffer];

      // Pre-processing: quantisation / rounding / zeroing (applied before compression)
      mCurrentTF.roundToIntegers(mRoundIntegersThreshold);
      if (mZeroThreshold > 0.f) {
        mCurrentTF.zeroSmallValues(mZeroThreshold);
      }
      if (mDynamicPrecisionSigma > 0.f) {
        mCurrentTF.trimGaussianPrecision(mDynamicPrecisionMean, mDynamicPrecisionSigma);
      }

      // Compress; the raw CMVPerTF branch is used when all flags are zero
      const uint8_t flags = buildCompressionFlags();
      if (flags != CMVEncoding::kNone) {
        mCurrentCompressedTF = mCurrentTF.compress(flags);
      }

      mIntervalTree->Fill();
      ++mIntervalTFCount;
      mCurrentTF = CMVPerTF{};
    }

    if (mProcessedTFs[currentBuffer] == mTimeFrames) {
      sendOutput(pc.outputs(), tf);
      finishInterval(pc, currentBuffer, tf);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "End of stream, flushing CMV interval ({} TFs)", mIntervalTFCount);
    // correct mTFEnd for the partial last interval so the CCDB validity end timestamp reflects the actual last TF, not the expected interval end
    mTFEnd[mBuffer] = mLastSeenTF;
    sendOutput(ec.outputs(), mLastSeenTF);
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
  bool mSendCCDB{false};                                                               ///< send output to CCDB populator
  bool mUsePreciseTimestamp{false};                                                    ///< use precise timestamp from orbit-reset info
  bool mDumpCMVs{false};                                                               ///< write a local ROOT debug file
  bool mUseCompressionVarint{false};                                                   ///< use delta+zigzag+varint compression (all values, no sparse skip); combined with mUseSparse → SparseV2 mode 1
  bool mUseSparse{false};                                                              ///< sparse encoding; alone = raw uint16 values; combined with varint/Huffman flag → SparseV2
  bool mUseCompressionHuffman{false};                                                  ///< Huffman encoding; combined with mUseSparse → SparseV2 mode 2
  uint16_t mRoundIntegersThreshold{0};                                                 ///< round values to nearest integer ADC for |v| <= N ADC; 0 = disabled
  float mZeroThreshold{0.f};                                                           ///< zero out CMV values whose float magnitude is below this threshold; 0 = disabled
  float mDynamicPrecisionMean{1.f};                                                    ///< Gaussian centre in |CMV| ADC where the strongest fractional-bit trimming is applied
  float mDynamicPrecisionSigma{0.f};                                                   ///< Gaussian width in ADC for the fractional-bit trimming; 0 disables
  long mTimestampStart{0};                                                             ///< CCDB validity start timestamp
  dataformats::Pair<long, int> mTFInfo{};                                              ///< orbit-reset time and NHBFPerTF for precise timestamp
  std::unique_ptr<TTree> mIntervalTree{};                                              ///< TTree accumulating one entry per completed TF in the current interval
  CMVPerTF mCurrentTF{};                                                               ///< staging object filled per CRU before compression
  CMVPerTFCompressed mCurrentCompressedTF{};                                           ///< compressed output for the current TF (used when flags != kNone)
  long mIntervalFirstTF{0};                                                            ///< absolute TF counter of the first TF in the current aggregation interval
  unsigned int mIntervalTFCount{0};                                                    ///< number of TTree entries filled for the current aggregation interval
  int mNFactorTFs{0};                                                                  ///< Number of TFs to skip for sending oldest TF
  int mNTFsDataDrop{0};                                                                ///< delay for the check if TFs are missing in TF units
  std::array<int, 2> mStartNTFsDataDrop{0};                                            ///< first relative TF to check
  long mProcessedTotalData{0};                                                         ///< used to check for dropeed TF data
  int mCheckEveryNData{1};                                                             ///< factor after which to check for missing data (in case data missing -> send dummy data)
  std::vector<InputSpec> mFilter{};                                                    ///< filter for looping over input data
  std::vector<InputSpec> mOrbitFilter{};                                               ///< filter for CMVORBITINFO from FLP
  std::array<std::vector<bool>, 2> mOrbitInfoForwarded{};                              ///< tracks whether orbit/BC has been captured per (buffer, relTF)
  uint32_t mLastSeenTF{0};                                                             ///< last TF counter seen in run(), used to set lastTF in endOfStream flush

  /// Returns real number of TFs taking buffer size into account
  unsigned int getNRealTFs() const { return mNTFsBuffer * mTimeFrames; }

  /// Build the CMVEncoding bitmask from the current option flags.
  uint8_t buildCompressionFlags() const
  {
    uint8_t flags = CMVEncoding::kNone;
    if (mUseSparse) {
      flags |= CMVEncoding::kSparse;
    }
    if (mUseCompressionHuffman) {
      flags |= CMVEncoding::kZigzag | CMVEncoding::kHuffman;
    } else if (mUseCompressionVarint) {
      flags |= CMVEncoding::kZigzag | CMVEncoding::kVarint;
    }
    // Delta coding is only applied for the dense (non-sparse) path with a value compressor
    if (!(flags & CMVEncoding::kSparse) && (flags & (CMVEncoding::kVarint | CMVEncoding::kHuffman))) {
      flags |= CMVEncoding::kDelta;
    }
    return flags;
  }

  /// Create a fresh in-memory TTree for the next aggregation interval.
  /// Uses a single CMVPerTFCompressed branch whenever any compression is active,
  /// or a raw CMVPerTF branch when no compression flags are set.
  void initIntervalTree()
  {
    mIntervalTree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
    mIntervalTree->SetAutoSave(0);
    mIntervalTree->SetDirectory(nullptr);
    if (buildCompressionFlags() != CMVEncoding::kNone) {
      mIntervalTree->Branch("CMVPerTFCompressed", &mCurrentCompressedTF);
    } else {
      mIntervalTree->Branch("CMVPerTF", &mCurrentTF);
    }
  }

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
      if (deviceProxy.getNumOutputChannels() > 0) {
        auto& state = deviceProxy.getOutputChannelState({0});
        size_t oldest = std::numeric_limits<size_t>::max() - 1; // just set to really large value
        state.oldestForChannel = {oldest};
      }
    }

    LOGP(info, "All TFs {} for current buffer received. Clearing buffer", tf);
    clearBuffer(buffer);
    mStartNTFsDataDrop[buffer] = 0;

    // reset per-interval state for the next aggregation interval
    initIntervalTree();
    mIntervalFirstTF = 0;
    mIntervalTFCount = 0;
    mCurrentTF = CMVPerTF{};
    mCurrentCompressedTF = CMVPerTFCompressed{};
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

    if (mIntervalTFCount == 0) {
      LOGP(warning, "CMV interval is empty at sendOutput, skipping");
      return;
    }

    // attach interval metadata to the TTree (stored once per tree)
    mIntervalTree->GetUserInfo()->Clear();
    mIntervalTree->GetUserInfo()->Add(new TParameter<long>("firstTF", mIntervalFirstTF));
    mIntervalTree->GetUserInfo()->Add(new TParameter<long>("lastTF", mLastSeenTF));

    LOGP(info, "CMVPerTF TTree: {} entries, firstTF={}, lastTF={}", mIntervalTFCount, mIntervalFirstTF, mLastSeenTF);
    auto start = timer::now();

    // write local ROOT file for debugging
    if (mDumpCMVs) {
      const std::string fname = fmt::format("CMV_timestamp{}.root", mTimestampStart);
      try {
        mCurrentTF.writeToFile(fname, mIntervalTree);
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
    // use the actual number of TFs in this interval (mIntervalTFCount) rather than mTimeFrames, so the CCDB validity end is correct for partial last intervals
    const long timeStampEnd = mTimestampStart + static_cast<long>(mIntervalTFCount * mNTFsBuffer * nHBFPerTF * o2::constants::lhc::LHCOrbitMUS * 1e-3);

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

    auto image = o2::ccdb::CcdbApi::createObjectImage((mIntervalTree.get()), &ccdbInfoCMV);
    // trim TMemFile zero-padding: GetSize() is block-rounded, GetEND() is the actual file end
    {
      TMemFile mf("trim", image->data(), static_cast<Long64_t>(image->size()), "READ");
      image->resize(static_cast<size_t>(mf.GetEND()));
      mf.Close();
    }
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
    Options{{"drop-data-after-nTFs", VariantType::Int, 0, {"Number of TFs after which to drop the data"}},
            {"check-data-every-n", VariantType::Int, 0, {"Number of run function called after which to check for missing data (-1 for no checking, 0 for default checking)"}},
            {"nFactorTFs", VariantType::Int, 1000, {"Number of TFs to skip for sending oldest TF"}},
            {"dump-cmvs", VariantType::Bool, false, {"Dump CMVs to a local ROOT file for debugging"}},
            {"use-sparse", VariantType::Bool, false, {"Sparse encoding (skip zero time bins). Alone: raw uint16 values. With --use-compression-varint: varint exact values. With --use-compression-huffman: Huffman exact values"}},
            {"use-compression-varint", VariantType::Bool, false, {"Delta+zigzag+varint compression (all values). Combined with --use-sparse: sparse positions + varint encoded exact CMV values"}},
            {"use-compression-huffman", VariantType::Bool, false, {"Huffman encoding. Combined with --use-sparse: sparse positions + Huffman-encoded exact CMV values"}},
            {"cmv-zero-threshold", VariantType::Float, 0.f, {"Zero out CMV values whose float magnitude is below this threshold after optional integer rounding and before compression; 0 disables"}},
            {"cmv-round-integers-threshold", VariantType::Int, 0, {"Round values to nearest integer ADC for |v| <= N ADC before compression; 0 disables"}},
            {"cmv-dynamic-precision-mean", VariantType::Float, 1.f, {"Gaussian centre in |CMV| ADC where the strongest fractional bit trimming is applied"}},
            {"cmv-dynamic-precision-sigma", VariantType::Float, 0.f, {"Gaussian width in ADC for smooth CMV fractional bit trimming; 0 disables"}}}}; // end DataProcessorSpec

  spec.rank = ilane;
  return spec;
}

} // namespace o2::tpc

#endif
