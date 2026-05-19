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

/// @file   TPCAggregateCMVSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  TPC aggregation of distributed CMVs, including preprocessing, compression and CCDB output

#ifndef O2_TPCAGGREGATECMVSPEC_H
#define O2_TPCAGGREGATECMVSPEC_H

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include "TMemFile.h"
#include "TParameter.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataTakingContext.h"
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonDataFormat/Pair.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "TPCWorkflow/TPCDistributeCMVSpec.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCCalibration/CMVContainer.h"
#include "DataFormatsTPC/CMV.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "MemoryResources/MemoryResources.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"

namespace o2::tpc
{

class TPCAggregateCMVDevice : public o2::framework::Task
{
 public:
  TPCAggregateCMVDevice(const int lane,
                        const std::vector<uint32_t>& crus,
                        const unsigned int timeframes,
                        const bool sendCCDB,
                        const bool usePreciseTimestamp,
                        const int nTFsBuffer,
                        std::shared_ptr<o2::base::GRPGeomRequest> req)
    : mLaneId{lane},
      mCRUs{crus},
      mTimeFrames{timeframes},
      mSendCCDB{sendCCDB},
      mUsePreciseTimestamp{usePreciseTimestamp},
      mNTFsBuffer{nTFsBuffer},
      mProcessedCRU(timeframes),
      mProcessedCRUs(timeframes),
      mRawCMVs(timeframes),
      mOrbitInfo(timeframes),
      mOrbitStep(timeframes),
      mOrbitInfoSeen(timeframes, false),
      mTFCompleted(timeframes, false),
      mCCDBRequest(req)
  {
    std::sort(mCRUs.begin(), mCRUs.end());
    for (auto& crusMap : mProcessedCRUs) {
      crusMap.reserve(mCRUs.size());
      for (const auto cruID : mCRUs) {
        crusMap.emplace(cruID, false);
      }
    }
    initIntervalTree();
  }

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mOutputDir = ic.options().get<std::string>("output-dir");
    if (mOutputDir != "/dev/null") {
      mOutputDir = o2::utils::Str::rectifyDirectory(mOutputDir);
    }
    mMetaFileDir = ic.options().get<std::string>("meta-output-dir");
    if (mMetaFileDir != "/dev/null") {
      mMetaFileDir = o2::utils::Str::rectifyDirectory(mMetaFileDir);
    }
    mUseCompressionVarint = ic.options().get<bool>("use-compression-varint");
    mUseSparse = ic.options().get<bool>("use-sparse");
    mUseCompressionHuffman = ic.options().get<bool>("use-compression-huffman");
    mRoundIntegersThreshold = static_cast<uint16_t>(ic.options().get<int>("cmv-round-integers-threshold"));
    mZeroThreshold = ic.options().get<float>("cmv-zero-threshold");
    mDynamicPrecisionMean = ic.options().get<float>("cmv-dynamic-precision-mean");
    mDynamicPrecisionSigma = ic.options().get<float>("cmv-dynamic-precision-sigma");
    mThreads = std::max(1, ic.options().get<int>("nthreads-compression"));
    LOGP(info, "CMV aggregation settings: output-dir={}, use-compression-varint={}, use-sparse={}, use-compression-huffman={}, cmv-round-integers-threshold={}, cmv-zero-threshold={}, cmv-dynamic-precision-mean={}, cmv-dynamic-precision-sigma={}, nthreads-compression={}",
         mOutputDir, mUseCompressionVarint, mUseSparse, mUseCompressionHuffman, mRoundIntegersThreshold, mZeroThreshold, mDynamicPrecisionMean, mDynamicPrecisionSigma, mThreads);
    initIntervalTree();
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // Consume CCDB inputs; return early when they are the only valid inputs in this slot
    int nCCDBInputs = 0;
    if (pc.inputs().isValid("grpecs")) {
      pc.inputs().get<o2::parameters::GRPECSObject*>("grpecs");
      ++nCCDBInputs;
    }
    if (mUsePreciseTimestamp && pc.inputs().isValid("orbitreset")) {
      mTFInfo = pc.inputs().get<dataformats::Pair<long, int>>("orbitreset");
      ++nCCDBInputs;
    }
    if (nCCDBInputs > 0 && pc.inputs().countValidInputs() == nCCDBInputs) {
      return;
    }

    if (mSetDataTakingCont) {
      mDataTakingContext = pc.services().get<o2::framework::DataTakingContext>();
      mSetDataTakingCont = false;
    }

    if (!mRun) {
      mRun = processing_helpers::getRunNumber(pc);
    }

    const auto currTF = processing_helpers::getCurrentTF(pc);

    if (mTFFirst == -1) {
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFirstTFFilter)) {
        mTFFirst = pc.inputs().get<long>(ref);
        mIntervalFirstTF = mTFFirst;
        mHasIntervalFirstTF = true;
        break;
      }
    }

    // EOS sentinel forwarded by the distribute lane for partial batches (n-TFs-buffer > actual TFs delivered)
    if (currTF == std::numeric_limits<uint32_t>::max()) {
      if (mTimestampStart == 0) {
        mTimestampStart = pc.services().get<o2::framework::TimingInfo>().creation;
      }
      collectEOSInputs(pc);
      return;
    }

    if (mTFFirst == -1) {
      mTFFirst = currTF;
      mIntervalFirstTF = mTFFirst;
      mHasIntervalFirstTF = true;
      LOGP(warning, "firstTF not found. Setting {} as first TF for aggregate lane {}", mTFFirst, mLaneId);
    }

    const long relTF = (currTF - mTFFirst) / mNTFsBuffer;
    if (relTF < 0) {
      LOGP(warning, "relTF={} < 0 for TF {}, skipping", relTF, currTF);
      return;
    }
    if (relTF >= static_cast<long>(mTimeFrames)) {
      // The distribute has advanced past this interval (empty CRU placeholders sent by checkMissingData
      // arrive with the triggering TF's context, not the missing batch's context).
      // Force-complete whatever was buffered so the next TF starts a fresh interval.
      LOGP(warning, "relTF={} out of range [0, {}) for TF {}: force-completing stale interval and resetting", relTF, mTimeFrames, currTF);
      if (mTimestampStart == 0) {
        mTimestampStart = static_cast<long>(pc.services().get<o2::framework::TimingInfo>().creation);
      }
      materializeBufferedTFs(true);
      sendOutput(pc.outputs());
      // Advance mTFFirst to the interval containing currTF so that after reset() clears it to -1
      // we can restore a valid value. Without this, the distribute won't resend CMVFIRSTTF (it was
      // already sent for the current interval), causing "firstTF not found" and further bad relTFs.
      long nextFirst = mIntervalFirstTF + static_cast<long>(mTimeFrames) * mNTFsBuffer;
      while (static_cast<long>(currTF) >= nextFirst + static_cast<long>(mTimeFrames) * mNTFsBuffer) {
        nextFirst += static_cast<long>(mTimeFrames) * mNTFsBuffer;
      }
      reset();
      mTFFirst = nextFirst;
      mIntervalFirstTF = nextFirst;
      mHasIntervalFirstTF = true;
      return;
    }

    // Capture orbit info first so setTimestampCCDB can use the measured stride
    if (!mOrbitInfoSeen[relTF]) {
      // all CRUs within a batch carry identical timing, so the first one is sufficient
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        mOrbitInfo[relTF] = pc.inputs().get<uint64_t>(ref);
        const auto batchFirstOrbit = static_cast<uint32_t>(mOrbitInfo[relTF] >> 32);
        // TimingInfo.firstTForbit is the orbit of the last real TF in the batch (the TF that triggered the FLP to send).
        // The FLP provides the orbit of the first real TF.  Interpolating between the two gives the true stride,
        // independent of the GRPECS/config nHBFPerTF value.
        const auto batchLastOrbit = static_cast<uint32_t>(pc.services().get<o2::framework::TimingInfo>().firstTForbit);
        const auto defaultOrbitStep = static_cast<uint32_t>(o2::base::GRPGeomHelper::instance().getNHBFPerTF());
        mOrbitStep[relTF] = ((batchFirstOrbit > 0) && (mNTFsBuffer > 1) && (batchLastOrbit > batchFirstOrbit)) ? (batchLastOrbit - batchFirstOrbit) / static_cast<uint32_t>(mNTFsBuffer - 1) : defaultOrbitStep;
        mLastOrbitStep = mOrbitStep[relTF];
        mOrbitInfoSeen[relTF] = true;
        break;
      }
    }

    if (mTimestampStart == 0) {
      setTimestampCCDB(relTF, mOrbitStep[relTF], pc);
    }

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = hdr->subSpecification;
      if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
        LOGP(debug, "Received CMV data from CRU {} which is not part of this aggregate lane", cru);
        continue;
      }
      if (mProcessedCRUs[relTF][cru]) {
        continue;
      }

      auto cmvVec = pc.inputs().get<o2::pmr::vector<uint16_t>>(ref);
      mRawCMVs[relTF][cru] = std::vector<uint16_t>(cmvVec.begin(), cmvVec.end());
      mProcessedCRUs[relTF][cru] = true;
      ++mProcessedCRU[relTF];
    }

    if (mProcessedCRU[relTF] == mCRUs.size() && !mTFCompleted[relTF]) {
      mTFCompleted[relTF] = true;
      ++mProcessedTFs;
      mLastSeenTF = currTF;
    }

    if (mProcessedTFs == mTimeFrames) {
      materializeBufferedTFs(false);
      sendOutput(pc.outputs());
      reset();
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    materializeBufferedTFs(true);
    materializeEOSBuffer();
    sendOutput(ec.outputs());
    ec.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionCCDBCMV() { return header::DataDescription{"TPC_CMV"}; }

 private:
  struct PreparedTF {
    CMVPerTF tf{};
    CMVPerTFCompressed compressed{};
  };

  const int mLaneId{0};                   ///< aggregate lane index (matches the distribute output lane)
  std::vector<uint32_t> mCRUs{};          ///< CRUs expected on this lane (sorted for binary_search)
  const unsigned int mTimeFrames{};       ///< number of CMV batches per calibration interval (= total TFs / nTFsBuffer)
  const bool mSendCCDB{false};            ///< send serialised TTree to the CCDB populator
  const bool mUsePreciseTimestamp{false}; ///< use orbit-reset info forwarded by the distribute lane for precise CCDB timestamps
  const int mNTFsBuffer{1};               ///< number of real TFs packed into one CMV batch (must match TPCFLPCMVSpec)
  std::string mOutputDir{};               ///< directory to write local ROOT files ("/dev/null" to disable)
  std::string mMetaFileDir{};             ///< directory to write calibration metadata files ("/dev/null" to disable)
  o2::framework::DataTakingContext mDataTakingContext{};
  bool mSetDataTakingCont{true};                                               ///< flag to capture DataTakingContext only once
  bool mUseCompressionVarint{false};                                           ///< delta+zigzag+varint compression for all values (dense path); combined with mUseSparse → sparse+varint
  bool mUseSparse{false};                                                      ///< sparse encoding (skip zero time bins); alone = raw uint16; combined with varint/Huffman → sparse+compressed
  bool mUseCompressionHuffman{false};                                          ///< Huffman encoding; combined with mUseSparse → sparse+Huffman
  uint16_t mRoundIntegersThreshold{0};                                         ///< round values to nearest integer ADC for |v| <= N ADC before compression; 0 = disabled
  float mZeroThreshold{0.f};                                                   ///< zero out CMV values whose float magnitude is below this threshold; 0 = disabled
  float mDynamicPrecisionMean{1.f};                                            ///< Gaussian centre in |CMV| ADC where the strongest fractional-bit trimming is applied
  float mDynamicPrecisionSigma{0.f};                                           ///< Gaussian width in ADC for fractional-bit trimming; 0 disables
  int mThreads{1};                                                             ///< number of threads for CMV preprocessing and compression in appendBatchToTree()
  long mTFFirst{-1};                                                           ///< absolute TF index of the first real TF in the current interval (-1 = not yet received)
  long mTimestampStart{0};                                                     ///< CCDB validity start timestamp in ms (0 until set by setTimestampCCDB)
  long mIntervalFirstTF{0};                                                    ///< absolute TF counter stored in the TTree UserInfo as "firstTF"
  bool mHasIntervalFirstTF{false};                                             ///< true once mIntervalFirstTF has been set for the current interval
  unsigned int mProcessedTFs{0};                                               ///< number of completed CMV batches in the current interval
  std::vector<unsigned int> mProcessedCRU{};                                   ///< counter of received CRUs per relTF slot; triggers completion when it reaches mCRUs.size()
  std::vector<std::unordered_map<unsigned int, bool>> mProcessedCRUs{};        ///< per-CRU received flag per relTF ([relTF][CRU]); prevents double-counting on retransmission
  std::vector<std::unordered_map<uint32_t, std::vector<uint16_t>>> mRawCMVs{}; ///< buffered raw CMV data per (relTF, CRU); unpacked in appendBatchToTree()
  std::vector<uint64_t> mOrbitInfo{};                                          ///< packed (firstOrbit << 32 | firstBC) per relTF, forwarded by the distribute lane
  std::vector<uint32_t> mOrbitStep{};                                          ///< per-sub-TF orbit stride per relTF; derived from actual batch timing
  std::vector<bool> mOrbitInfoSeen{};                                          ///< true once orbit/BC has been captured for each relTF slot
  std::vector<bool> mTFCompleted{};                                            ///< true once all CRUs have been received for a given relTF slot
  std::unordered_map<uint32_t, std::vector<uint16_t>> mEOSRawCMVs{};           ///< CMV data received during the EOS sentinel path (partial batch at end of run)
  uint32_t mEOSFirstOrbit{0};                                                  ///< firstOrbit captured from the FLP's EOS partial-buffer flush
  uint16_t mEOSFirstBC{0};                                                     ///< firstBC captured from the FLP's EOS partial-buffer flush
  uint32_t mLastOrbitStep{0};                                                  ///< cached orbit stride from the last complete batch; fallback for the EOS partial batch
  uint32_t mLastSeenTF{0};                                                     ///< last TF counter seen in run(); used to compute lastTF metadata in the TTree
  unsigned int mIntervalTFCount{0};                                            ///< number of TTree entries filled for the current interval
  uint64_t mRun{0};                                                            ///< run number, captured once per run
  uint32_t mIntervalFirstOrbit{0};                                             ///< first orbit of the first TF in the current interval
  uint32_t mIntervalLastOrbit{0};                                              ///< first orbit of the last TF in the current interval
  uint32_t mFirstOrbitDPL{0};                                                  ///< first orbit of the first TF in the current interval
  bool mIntervalOrbitSet{false};                                               ///< true once first orbit has been captured for the current interval
  dataformats::Pair<long, int> mTFInfo{};                                      ///< orbit-reset time (ms) and NHBFPerTF forwarded by distribute lane 0 for precise timestamps
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                      ///< GRPECS request so GRPGeomHelper::getNHBFPerTF() is valid in this process
  std::unique_ptr<TTree> mIntervalTree{};                                      ///< in-memory TTree accumulating one entry per real TF; serialised to CCDB/disk at interval end
  CMVPerTF mCurrentTF{};                                                       ///< staging object written to the TTree branch for the uncompressed path
  CMVPerTFCompressed mCurrentCompressedTF{};                                   ///< staging object written to the TTree branch when any compression flags are set
  const std::vector<o2::framework::InputSpec> mFilter{
    {"cmvagg",
     o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(mLaneId)},
     o2::framework::Lifetime::Sporadic}};
  const std::vector<o2::framework::InputSpec> mOrbitFilter{
    {"cmvorbit",
     o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitInfo(mLaneId), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(mLaneId)}},
     o2::framework::Lifetime::Sporadic}};
  const std::vector<o2::framework::InputSpec> mFirstTFFilter{
    {"firstTF",
     o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVFirstTF(), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(mLaneId)}},
     o2::framework::Lifetime::Sporadic}};

  uint8_t buildCompressionFlags() const
  {
    uint8_t flags = CMVEncoding::kNone;
    if (mUseSparse) {
      flags |= CMVEncoding::kSparse;
    }
    if (mUseCompressionHuffman) {
      flags |= CMVEncoding::kDelta | CMVEncoding::kZigzag | CMVEncoding::kHuffman;
    } else if (mUseCompressionVarint) {
      flags |= CMVEncoding::kDelta | CMVEncoding::kZigzag | CMVEncoding::kVarint;
    }
    return flags;
  }

  /// Create a fresh in-memory TTree for the next aggregation interval
  /// Uses a single CMVPerTFCompressed branch whenever any compression is active or a raw CMVPerTF branch when no compression flags are set.
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

  /// Accumulate CMV data from the EOS sentinel (TF == UINT32_MAX), i.e. a partial batch forwarded by the distribute lane when n-TFs-buffer > number of TFs actually delivered
  /// Orbit/BC is captured once; raw data is appended per CRU into mEOSRawCMVs
  void collectEOSInputs(o2::framework::ProcessingContext& pc)
  {
    if (mEOSFirstOrbit == 0) {
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        const auto orbitBC = pc.inputs().get<uint64_t>(ref);
        mEOSFirstOrbit = static_cast<uint32_t>(orbitBC >> 32);
        mEOSFirstBC = static_cast<uint16_t>(orbitBC & 0xFFFFu);
        break;
      }
    }

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const unsigned int cru = hdr->subSpecification;
      if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
        continue;
      }
      auto cmvVec = pc.inputs().get<o2::pmr::vector<uint16_t>>(ref);
      auto& buffer = mEOSRawCMVs[cru];
      buffer.insert(buffer.end(), cmvVec.begin(), cmvVec.end());
    }
  }

  /// Set the CCDB validity start timestamp
  /// When using precise timestamps, back-calculates the orbit-reset-referenced wall-clock time for the first real TF in the interval using the orbit-reset time forwarded by distribute lane 0.
  /// orbitStep is the dynamically measured per-sub-TF stride; when non-zero it is preferred over the GRP NHBFPerTF for the orbit-offset calculation.
  void setTimestampCCDB(const long relTF, const uint32_t orbitStep, o2::framework::ProcessingContext& pc)
  {
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (mUsePreciseTimestamp && !mTFInfo.second) {
      // Orbit-reset info (NHBFPerTF) not yet received from the distribute lane.
      // Fall back to DPL wall-clock creation time so mTimestampStart is never
      // left at 0, which would cause successive intervals to overwrite each other.
      mTimestampStart = tinfo.creation;
      LOGP(warning, "Orbit reset info not yet received; using DPL creation time {} ms as fallback timestamp for interval starting at TF {}", mTimestampStart, mTFFirst);
      return;
    }
    // prefer the measured stride; fall back to NHBFPerTF from GRPECS
    const int nHBFPerTF = (orbitStep > 0) ? static_cast<int>(orbitStep) : o2::base::GRPGeomHelper::instance().getNHBFPerTF();
    const auto nOrbitsOffset = (relTF * mNTFsBuffer + (mNTFsBuffer - 1)) * nHBFPerTF;
    mFirstOrbitDPL = tinfo.firstTForbit - nOrbitsOffset;
    mTimestampStart = mUsePreciseTimestamp ? (mTFInfo.first + (tinfo.firstTForbit - nOrbitsOffset) * o2::constants::lhc::LHCOrbitMUS * 0.001) : tinfo.creation;
    LOGP(info, "Setting timestamp reset reference to: {}, at tfCounter: {}, firstTForbit: {}, NHBFPerTF: {}, relTF: {}, nOrbitsOffset: {}",
         mTFInfo.first, tinfo.tfCounter, tinfo.firstTForbit, nHBFPerTF, relTF, nOrbitsOffset);
  }

  /// Unpack and fill the TTree for all relTF slots that have been buffered during run().
  /// When includeIncomplete=false (normal interval end) only fully-received batches are filled.
  /// When includeIncomplete=true (EOS flush) partial batches are also flushed with a warning.
  void materializeBufferedTFs(const bool includeIncomplete)
  {
    for (unsigned int relTF = 0; relTF < mTimeFrames; ++relTF) {
      if (mProcessedCRU[relTF] == 0) {
        continue;
      }

      if ((mProcessedCRU[relTF] != mCRUs.size()) && !includeIncomplete) {
        continue;
      }

      if ((mProcessedCRU[relTF] != mCRUs.size()) && includeIncomplete) {
        LOGP(warning, "Aggregate lane {} flushing incomplete CMV batch relTF {} at EOS: received {} CRUs out of {}", mLaneId, relTF, mProcessedCRU[relTF], mCRUs.size());
      }

      if (!mHasIntervalFirstTF) {
        mIntervalFirstTF = mTFFirst == -1 ? 0 : mTFFirst;
        mHasIntervalFirstTF = true;
      }

      // derive the actual number of sub-TFs from the buffer size; fall back to mNTFsBuffer if empty
      const auto maxBufferSize = getMaxBufferSize(mRawCMVs[relTF]);
      const int nTFsInBatch = maxBufferSize ? std::max(1, static_cast<int>(maxBufferSize / cmv::NTimeBinsPerTF)) : mNTFsBuffer;
      // fall back to GRP NHBFPerTF only if no orbit stride was measured for this relTF
      const auto orbitStep = mOrbitStep[relTF] ? mOrbitStep[relTF] : static_cast<uint32_t>(o2::base::GRPGeomHelper::instance().getNHBFPerTF());
      appendBatchToTree(mRawCMVs[relTF], mOrbitInfo[relTF], orbitStep, nTFsInBatch);
    }
  }

  /// Unpack and fill the TTree from the EOS partial-batch buffer (mEOSRawCMVs).
  /// The number of real TFs is inferred from the raw buffer size divided by NTimeBinsPerTF.
  /// Uses mLastOrbitStep from the last complete batch as the orbit stride fallback.
  void materializeEOSBuffer()
  {
    if (mEOSRawCMVs.empty()) {
      return;
    }

    const auto maxBufferSize = getMaxBufferSize(mEOSRawCMVs);
    const int nTFsInBatch = static_cast<int>(maxBufferSize / cmv::NTimeBinsPerTF);
    if (nTFsInBatch <= 0) {
      return;
    }

    if (!mHasIntervalFirstTF) {
      mIntervalFirstTF = mLastSeenTF + 1;
      mHasIntervalFirstTF = true;
    }

    const uint64_t orbitInfo = (static_cast<uint64_t>(mEOSFirstOrbit) << 32) | static_cast<uint64_t>(mEOSFirstBC);
    // use the actual stride seen in run(); fall back to GRP only if no complete batch was seen
    const auto orbitStep = mLastOrbitStep ? mLastOrbitStep : static_cast<uint32_t>(o2::base::GRPGeomHelper::instance().getNHBFPerTF());
    appendBatchToTree(mEOSRawCMVs, orbitInfo, orbitStep, nTFsInBatch);
    mLastSeenTF += static_cast<uint32_t>(nTFsInBatch);
  }

  static size_t getMaxBufferSize(const std::unordered_map<uint32_t, std::vector<uint16_t>>& rawCMVs)
  {
    size_t maxBufferSize = 0;
    for (const auto& [cru, values] : rawCMVs) {
      maxBufferSize = std::max(maxBufferSize, values.size());
    }
    return maxBufferSize;
  }

  /// Unpack nTFsInBatch real TFs from rawCMVs, apply preprocessing (rounding, zeroing, trimming),
  /// optionally compress them, and fill one TTree entry per real TF.
  /// Processing is parallelised across nThreads workers using std::thread (each thread owns a disjoint chunk).
  void appendBatchToTree(const std::unordered_map<uint32_t, std::vector<uint16_t>>& rawCMVs, const uint64_t orbitInfo, const uint32_t orbitStep, const int nTFsInBatch)
  {
    if (nTFsInBatch <= 0) {
      return;
    }

    const auto firstOrbit = static_cast<uint32_t>(orbitInfo >> 32);
    const auto firstBC = static_cast<uint16_t>(orbitInfo & 0xFFFFu);
    // Use the DPL-derived orbit as fallback when the FLP orbit info is missing (firstOrbit == 0)
    const auto batchFirstOrbitDPL = (firstOrbit > 0) ? firstOrbit : mFirstOrbitDPL;
    if (!mIntervalOrbitSet) {
      mIntervalFirstOrbit = batchFirstOrbitDPL;
      mIntervalOrbitSet = true;
    }
    mIntervalLastOrbit = batchFirstOrbitDPL + static_cast<uint32_t>(nTFsInBatch - 1) * orbitStep;
    const uint8_t flags = buildCompressionFlags();
    std::vector<PreparedTF> prepared(nTFsInBatch);
    const int nThreads = std::max(1, std::min(mThreads, nTFsInBatch));
    const int chunkSize = (nTFsInBatch + nThreads - 1) / nThreads;

    auto worker = [&](const int iThread) {
      const int beginTF = iThread * chunkSize;
      const int endTF = std::min(nTFsInBatch, beginTF + chunkSize);
      for (int tfIndex = beginTF; tfIndex < endTF; ++tfIndex) {

        auto& preparedTF = prepared[tfIndex];
        preparedTF.tf.firstOrbit = firstOrbit + static_cast<uint32_t>(tfIndex) * orbitStep;
        preparedTF.tf.firstOrbitDPL = batchFirstOrbitDPL + static_cast<uint32_t>(tfIndex) * orbitStep;

        for (const auto& [cru, values] : rawCMVs) {
          const uint32_t offset = static_cast<uint32_t>(tfIndex) * cmv::NTimeBinsPerTF;
          if (offset >= static_cast<uint32_t>(values.size())) {
            continue;
          }
          const uint32_t nBins = std::min(static_cast<uint32_t>(values.size()) - offset, cmv::NTimeBinsPerTF);
          for (uint32_t tb = 0; tb < nBins; ++tb) {
            preparedTF.tf.mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] = values[offset + tb];
          }
        }

        preparedTF.tf.roundToIntegers(mRoundIntegersThreshold);
        if (mZeroThreshold > 0.f) {
          preparedTF.tf.zeroSmallValues(mZeroThreshold);
        }
        if (mDynamicPrecisionSigma > 0.f) {
          preparedTF.tf.trimGaussianPrecision(mDynamicPrecisionMean, mDynamicPrecisionSigma);
        }
        if (flags != CMVEncoding::kNone) {
          preparedTF.compressed = preparedTF.tf.compress(flags);
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(nThreads - 1);
    for (int iThread = 1; iThread < nThreads; ++iThread) {
      workers.emplace_back(worker, iThread);
    }
    worker(0);
    for (auto& thread : workers) {
      thread.join();
    }

    for (int tfIndex = 0; tfIndex < nTFsInBatch; ++tfIndex) {
      if (flags != CMVEncoding::kNone) {
        mCurrentCompressedTF = std::move(prepared[tfIndex].compressed);
      } else {
        mCurrentTF = std::move(prepared[tfIndex].tf);
      }
      mIntervalTree->Fill();
      ++mIntervalTFCount;
    }
  }

  void sendOutput(o2::framework::DataAllocator& output)
  {
    using timer = std::chrono::high_resolution_clock;

    if (mIntervalTFCount == 0) {
      LOGP(warning, "CMV interval is empty at sendOutput for lane {}, skipping", mLaneId);
      return;
    }

    const auto lastTF = mIntervalFirstTF + static_cast<long>(mIntervalTFCount) - 1;
    mIntervalTree->GetUserInfo()->Clear();
    mIntervalTree->GetUserInfo()->Add(new TParameter<long>("firstTF", mIntervalFirstTF));
    mIntervalTree->GetUserInfo()->Add(new TParameter<long>("lastTF", lastTF));

    LOGP(info, "CMVPerTF TTree lane {}: {} entries, firstTF={}, lastTF={}", mLaneId, mIntervalTFCount, mIntervalFirstTF, lastTF);
    auto start = timer::now();

    const int nHBFPerTF = o2::base::GRPGeomHelper::instance().getNHBFPerTF();
    const long timeStampEnd = mTimestampStart + static_cast<long>(mIntervalTFCount * nHBFPerTF * o2::constants::lhc::LHCOrbitMUS * 1e-3);

    if (mOutputDir != "/dev/null") {
      const std::string calibFName = fmt::format("CMV_run_{}_orbit_{}_{}_timestamp_{}_{}.root",
                                                 mRun, mIntervalFirstOrbit, mIntervalLastOrbit, mTimestampStart, timeStampEnd);
      try {
        CMVPerTF::writeToFile(mOutputDir + calibFName, mIntervalTree);
        LOGP(info, "CMV file written to {}", mOutputDir + calibFName);
      } catch (const std::exception& e) {
        LOGP(error, "Failed to write CMV file {}: {}", mOutputDir + calibFName, e.what());
      }

      if (mMetaFileDir != "/dev/null") {
        o2::dataformats::FileMetaData calMetaData;
        calMetaData.fillFileData(mOutputDir + calibFName);
        calMetaData.setDataTakingContext(mDataTakingContext);
        calMetaData.type = "calib";
        calMetaData.priority = "low";
        auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaFileDir, calibFName);
        auto metaFileName = fmt::format("{}{}.done", mMetaFileDir, calibFName);
        try {
          std::ofstream metaFileOut(metaFileNameTmp);
          metaFileOut << calMetaData;
          metaFileOut.close();
          std::filesystem::rename(metaFileNameTmp, metaFileName);
        } catch (std::exception const& e) {
          LOG(error) << "Failed to store CMV meta data file " << metaFileName << ", reason: " << e.what();
        }
      }
    }

    if ((!mSendCCDB) && (mOutputDir == "/dev/null")) {
      LOGP(warning, "Neither CCDB output nor output-dir is enabled for aggregate lane {}, skipping CMV export", mLaneId);
    }
    if (!mSendCCDB) {
      return;
    }

    if (timeStampEnd <= mTimestampStart) {
      LOGP(warning, "Invalid CCDB timestamp range start:{} end:{}, skipping upload", mTimestampStart, timeStampEnd);
      return;
    }

    o2::ccdb::CcdbObjectInfo ccdbInfoCMV("TPC/Calib/CMV", "TTree", "CMV.root", {}, mTimestampStart, timeStampEnd);
    auto image = o2::ccdb::CcdbApi::createObjectImage((mIntervalTree.get()), &ccdbInfoCMV);
    // trim TMemFile zero-padding: GetSize() is block-rounded, GetEND() is the actual file end
    {
      TMemFile mf("trim", image->data(), static_cast<Long64_t>(image->size()), "READ");
      image->resize(static_cast<size_t>(mf.GetEND()));
      mf.Close();
    }

    LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {}", ccdbInfoCMV.getPath(), ccdbInfoCMV.getFileName(), image->size(), ccdbInfoCMV.getStartValidityTimestamp(), ccdbInfoCMV.getEndValidityTimestamp());
    output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBCMV(), 0}, *image);
    output.snapshot(o2::framework::Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBCMV(), 0}, ccdbInfoCMV);

    auto stop = timer::now();
    std::chrono::duration<float> elapsed = stop - start;
    LOGP(info, "CMV CCDB serialisation time: {:.3f} s", elapsed.count());
  }

  /// Reset all per-interval state after a successful sendOutput(); prepares for the next interval
  void reset()
  {
    mTFFirst = -1;
    mTimestampStart = 0;
    mIntervalFirstTF = 0;
    mHasIntervalFirstTF = false;
    mProcessedTFs = 0;
    std::fill(mProcessedCRU.begin(), mProcessedCRU.end(), 0);
    std::fill(mOrbitInfo.begin(), mOrbitInfo.end(), 0);
    std::fill(mOrbitStep.begin(), mOrbitStep.end(), 0);
    std::fill(mOrbitInfoSeen.begin(), mOrbitInfoSeen.end(), false);
    std::fill(mTFCompleted.begin(), mTFCompleted.end(), false);
    for (auto& processedMap : mProcessedCRUs) {
      for (auto& [cru, seen] : processedMap) {
        seen = false;
      }
    }
    for (auto& rawPerTF : mRawCMVs) {
      rawPerTF.clear();
    }
    mEOSRawCMVs.clear();
    mEOSFirstOrbit = 0;
    mEOSFirstBC = 0;
    mLastOrbitStep = 0;
    mLastSeenTF = 0;
    mIntervalTFCount = 0;
    mIntervalFirstOrbit = 0;
    mIntervalLastOrbit = 0;
    mFirstOrbitDPL = 0;
    mIntervalOrbitSet = false;
    mCurrentTF = CMVPerTF{};
    mCurrentCompressedTF = CMVPerTFCompressed{};
    initIntervalTree();
  }
};

/// Build a DataProcessorSpec for one aggregate lane
/// Each lane receives CMV data from one distribute output lane (matched by lane index) and expects the full CRU list — the distribute stage already routes per-CRU data to the correct lane
inline o2::framework::DataProcessorSpec getTPCAggregateCMVSpec(const int lane,
                                                               const std::vector<uint32_t>& crus,
                                                               const unsigned int timeframes,
                                                               const bool sendCCDB,
                                                               const bool usePreciseTimestamp,
                                                               const int nTFsBuffer = 1)
{
  std::vector<o2::framework::OutputSpec> outputSpecs;
  if (sendCCDB) {
    outputSpecs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCAggregateCMVDevice::getDataDescriptionCCDBCMV()}, o2::framework::Lifetime::Sporadic);
    outputSpecs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCAggregateCMVDevice::getDataDescriptionCCDBCMV()}, o2::framework::Lifetime::Sporadic);
  }

  std::vector<o2::framework::InputSpec> inputSpecs;
  inputSpecs.emplace_back(o2::framework::InputSpec{"cmvagg", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(lane)}, o2::framework::Lifetime::Sporadic});
  inputSpecs.emplace_back(o2::framework::InputSpec{"cmvorbit", o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitInfo(lane), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)}, o2::framework::Lifetime::Sporadic});
  inputSpecs.emplace_back(o2::framework::InputSpec{"firstTF", o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVFirstTF(), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)}, o2::framework::Lifetime::Sporadic});
  if (usePreciseTimestamp) {
    inputSpecs.emplace_back(o2::framework::InputSpec{"orbitreset", o2::header::gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitReset(), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)}, o2::framework::Lifetime::Sporadic});
  }

  // Request GRPECS from CCDB so that GRPGeomHelper::getNHBFPerTF() is valid in this (separate) process
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS (NHBFPerTF)
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputSpecs);

  o2::framework::DataProcessorSpec spec{
    fmt::format("tpc-aggregate-cmv-{:02}", lane).data(),
    inputSpecs,
    outputSpecs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<TPCAggregateCMVDevice>(lane, crus, timeframes, sendCCDB, usePreciseTimestamp, nTFsBuffer, ccdbRequest)},
    o2::framework::Options{{"output-dir", o2::framework::VariantType::String, "/dev/null", {"CMV output directory, must exist (if not /dev/null)"}},
                           {"meta-output-dir", o2::framework::VariantType::String, "/dev/null", {"calibration metadata output directory, must exist (if not /dev/null)"}},
                           {"nthreads-compression", o2::framework::VariantType::Int, 1, {"Number of threads used for CMV per timeframe preprocessing and compression"}},
                           {"use-sparse", o2::framework::VariantType::Bool, false, {"Sparse encoding (skip zero time bins). Alone: raw uint16 values. With --use-compression-varint: varint exact values. With --use-compression-huffman: Huffman exact values"}},
                           {"use-compression-varint", o2::framework::VariantType::Bool, false, {"Delta+zigzag+varint compression (all values). Combined with --use-sparse: sparse positions + varint encoded exact CMV values"}},
                           {"use-compression-huffman", o2::framework::VariantType::Bool, false, {"Huffman encoding. Combined with --use-sparse: sparse positions + Huffman-encoded exact CMV values"}},
                           {"cmv-zero-threshold", o2::framework::VariantType::Float, 0.f, {"Zero out CMV values whose float magnitude is below this threshold after optional integer rounding and before compression; 0 disables"}},
                           {"cmv-round-integers-threshold", o2::framework::VariantType::Int, 0, {"Round values to nearest integer ADC for |v| <= N ADC before compression; 0 disables"}},
                           {"cmv-dynamic-precision-mean", o2::framework::VariantType::Float, 1.f, {"Gaussian centre in |CMV| ADC where the strongest fractional bit trimming is applied"}},
                           {"cmv-dynamic-precision-sigma", o2::framework::VariantType::Float, 0.f, {"Gaussian width in ADC for smooth CMV fractional bit trimming; 0 disables"}}}};
  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif
