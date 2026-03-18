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

/// @file   TPCFactorizeCMVSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  TPC device that collects per CRU CMV vectors over an aggregation interval and writes them as a TTree (via CMVContainer) to the CCDB

#ifndef O2_TPCFACTORIZECMVSPEC_H
#define O2_TPCFACTORIZECMVSPEC_H

#include <vector>
#include <chrono>
#include <fmt/format.h>

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataTakingContext.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/Pair.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "TPCWorkflow/TPCDistributeCMVSpec.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCCalibration/CMVContainer.h"
#include "DataFormatsTPC/CMV.h"
#include "MemoryResources/MemoryResources.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;

namespace o2::tpc
{

class TPCFactorizeCMVDevice : public o2::framework::Task
{
 public:
  TPCFactorizeCMVDevice(const int lane,
                        const std::vector<uint32_t>& crus,
                        const unsigned int timeframes,
                        const bool sendCCDB,
                        const bool usePreciseTimestamp,
                        const int nTFsBuffer)
    : mLaneId{lane},
      mCRUs{crus},
      mTimeFrames{timeframes},
      mSendCCDB{sendCCDB},
      mUsePreciseTimestamp{usePreciseTimestamp},
      mNTFsBuffer{nTFsBuffer}
  {
    // Pre-allocate
    mInterval.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
  }

  void init(o2::framework::InitContext& ic) final
  {
    mDumpCMVs = ic.options().get<bool>("dump-cmvs");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // Capture orbit-reset info once for precise CCDB timestamp calculation
    if (mUsePreciseTimestamp && pc.inputs().isValid("orbitreset")) {
      mTFInfo = pc.inputs().get<dataformats::Pair<long, int>>("orbitreset");
      if (pc.inputs().countValidInputs() == 1) {
        return; // only the orbit-reset message arrived this round
      }
    }

    // Record the absolute first TF of this aggregation interval
    const auto currTF = processing_helpers::getCurrentTF(pc);

    if (mTFFirst == -1 && pc.inputs().isValid("firstTF")) {
      mTFFirst = pc.inputs().get<long>("firstTF");
      mInterval.firstTF = mTFFirst;
    }
    if (mTFFirst == -1) {
      mTFFirst = currTF;
      mInterval.firstTF = mTFFirst;
      LOGP(warning, "firstTF not found! Setting {} as first TF", mTFFirst);
    }

    // Set data taking context only once
    if (mSetDataTakingCont) {
      mDataTakingContext = pc.services().get<DataTakingContext>();
      mSetDataTakingCont = false;
    }

    // Set the run number only once
    if (!mRun) {
      mRun = processing_helpers::getRunNumber(pc);
    }

    const long relTF = (currTF - mTFFirst) / mNTFsBuffer;

    // Set CCDB start timestamp once at the start of each aggregation interval
    if (mTimestampStart == 0) {
      setTimestampCCDB(relTF, pc);
    }

    if (relTF < 0 || relTF >= static_cast<long>(mTimeFrames)) {
      LOGP(warning, "relTF={} out of range [0, {}), skipping TF {}", relTF, mTimeFrames, currTF);
      return;
    }

    // Apply orbit/BC info for any relTF whose CMVORBITINFO message has arrived
    for (auto& ref : InputRecordWalker(pc.inputs(), mOrbitFilter)) {
      auto const* hdr = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const uint32_t orbitRelTF = static_cast<uint32_t>(hdr->subSpecification);
      if (orbitRelTF < mTimeFrames) {
        auto& tfData = mInterval.mCMVPerTF[orbitRelTF];
        if (tfData.firstOrbit == 0 && tfData.firstBC == 0) {
          const auto orbitBC = pc.inputs().get<uint64_t>(ref);
          tfData.firstOrbit = static_cast<int64_t>(orbitBC >> 32);
          tfData.firstBC = static_cast<int64_t>(orbitBC & 0xFFFFu);
        }
      }
    }

    // Consume all incoming CMV vectors for this TF
    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* hdr = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const uint32_t cru = hdr->subSpecification;

      auto cmvVec = pc.inputs().get<pmr::vector<float>>(ref);
      if (cmvVec.empty()) {
        LOGP(warning, "Received empty CMV vector for CRU {}, skipping", cru);
        ++mProcessedCRUs; // count it to avoid stalling the completion check
        continue;
      }

      // Each TF carries 4 packets × NTimeBinsPerPacket floats = NTimeBinsPerTF floats
      if (cmvVec.size() != cmv::NTimeBinsPerTF) {
        LOGP(warning, "CRU {}: got {} CMV values, expected {} (4 packets × {})",
             cru, cmvVec.size(), cmv::NTimeBinsPerTF, cmv::NTimeBinsPerPacket);
      }
      mInterval.mCMVPerTF[relTF].mDataPerTF[cru].assign(cmvVec.begin(), cmvVec.end());
      ++mProcessedCRUs;
    }

    // Once all CRUs × all TFs have been received, write out
    if (mProcessedCRUs == mCRUs.size() * mTimeFrames) {
      mInterval.lastTF = currTF;
      LOGP(info, "ProcessedTFs: {}  currTF: {}  relTF: {}  OrbitResetTime: {}  orbits per TF: {}",
           mProcessedCRUs / mCRUs.size(), currTF, relTF, mTFInfo.first, mTFInfo.second);
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "End of stream, flushing CMV interval ({} TFs, lane {})", mInterval.size(), mLaneId);
    sendOutput(ec.outputs());
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionCCDBCMV() { return header::DataDescription{"TPC_CMV"}; }

 private:
  const int mLaneId{0};
  const std::vector<uint32_t> mCRUs{};
  const unsigned int mTimeFrames{};
  const bool mSendCCDB{false};
  const bool mUsePreciseTimestamp{false};
  const int mNTFsBuffer{1};
  bool mDumpCMVs{false}; ///< write a local ROOT debug file
  long mTFFirst{-1};
  long mTimestampStart{0};
  unsigned int mProcessedCRUs{0}; ///< total CRU entries received in this interval
  uint64_t mRun{0};
  dataformats::Pair<long, int> mTFInfo{};
  CMVPerInterval mInterval{};
  o2::framework::DataTakingContext mDataTakingContext{};
  bool mSetDataTakingCont{true};
  const std::vector<InputSpec> mFilter{
    {"cmvagg",
     ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(mLaneId)},
     Lifetime::Sporadic}};

  /// Filter for per-TF orbit/BC info from the distribute device
  const std::vector<InputSpec> mOrbitFilter{
    {"orbitinfo",
     ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitInfo(mLaneId)},
     Lifetime::Sporadic}};

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

  void sendOutput(DataAllocator& output)
  {
    using timer = std::chrono::high_resolution_clock;

    if (mInterval.empty()) {
      LOGP(warning, "CMV interval is empty at sendOutput (lane {}), skipping", mLaneId);
      reset();
      return;
    }

    // Check if any CRU actually wrote data into this interval
    const bool hasData = std::any_of(mInterval.mCMVPerTF.begin(), mInterval.mCMVPerTF.end(),
                                     [](const CMVPerTF& tf) {
                                       return std::any_of(tf.mDataPerTF.begin(), tf.mDataPerTF.end(),
                                                          [](const std::vector<float>& v) { return !v.empty(); });
                                     });
    if (!hasData) {
      LOGP(warning, "CMV interval has no data at sendOutput (lane {}), skipping", mLaneId);
      reset();
      return;
    }

    LOGP(info, "{}", mInterval.summary());
    auto start = timer::now();
    auto tree = mInterval.toTTree();

    // Write local ROOT file for debugging
    if (mDumpCMVs) {
      const std::string fname = fmt::format("CMV_lane{:02}_timestamp{}.root", mLaneId, mTimestampStart);
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

    reset();
  }

  /// Reset all per-interval state
  void reset()
  {
    mInterval.clear();
    mInterval.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
    mTimestampStart = 0;
    mTFFirst = -1;
    mProcessedCRUs = 0;
    mSetDataTakingCont = true;
    LOGP(info, "Everything cleared. Waiting for new data to arrive.");
  }
};

inline DataProcessorSpec getTPCFactorizeCMVSpec(
  const int lane,
  const std::vector<uint32_t>& crus,
  const unsigned int timeframes,
  const bool sendCCDB,
  const bool usePreciseTimestamp,
  const int nTFsBuffer = 1)
{
  std::vector<OutputSpec> outputSpecs;
  if (sendCCDB) {
    outputSpecs.emplace_back(
      ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload,
                              TPCFactorizeCMVDevice::getDataDescriptionCCDBCMV()},
      Lifetime::Sporadic);
    outputSpecs.emplace_back(
      ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper,
                              TPCFactorizeCMVDevice::getDataDescriptionCCDBCMV()},
      Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputSpecs;
  // CMV float vectors from the distribute device, one per CRU per TF
  inputSpecs.emplace_back(InputSpec{
    "cmvagg",
    ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(lane)},
    Lifetime::Sporadic});
  // Per-TF orbit/BC info from the distribute device (subSpecification == relTF)
  inputSpecs.emplace_back(InputSpec{
    "orbitinfo",
    ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMVOrbitInfo(lane)},
    Lifetime::Sporadic});
  // First TF of the current aggregation interval
  inputSpecs.emplace_back(InputSpec{
    "firstTF",
    gDataOriginTPC,
    TPCDistributeCMVSpec::getDataDescriptionCMVFirstTF(),
    header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)},
    Lifetime::Sporadic});
  if (usePreciseTimestamp) {
    inputSpecs.emplace_back(InputSpec{
      "orbitreset",
      gDataOriginTPC,
      TPCDistributeCMVSpec::getDataDescriptionCMVOrbitReset(),
      header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)},
      Lifetime::Sporadic});
  }

  const std::string type = "cmv";
  DataProcessorSpec spec{
    fmt::format("tpc-factorize-{}-{:02}", type, lane).data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFactorizeCMVDevice>(lane, crus, timeframes, sendCCDB, usePreciseTimestamp, nTFsBuffer)},
    Options{
      {"dump-cmvs", VariantType::Bool, false, {"Dump CMVs to a local ROOT file for debugging"}}}};

  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif // O2_TPCFACTORIZECMVSPEC_H