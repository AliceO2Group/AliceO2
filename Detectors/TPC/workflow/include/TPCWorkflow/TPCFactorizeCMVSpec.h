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
    mContainer.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
  }

  void init(o2::framework::InitContext& ic) final
  {
    mNOrbitsCMV = ic.options().get<int>("orbits-CMVs");
    mDumpCMVs = ic.options().get<bool>("dump-cmvs");
    mOffsetCCDB = ic.options().get<bool>("add-offset-for-CCDB-timestamp");
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
      mContainer.firstTF = mTFFirst;
    }
    if (mTFFirst == -1) {
      mTFFirst = currTF;
      mContainer.firstTF = mTFFirst;
      LOGP(warning, "firstTF not Found! Found valid inputs {}. Setting {} as first TF", pc.inputs().countValidInputs(), mTFFirst);
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

      // Store one entry per timebin into the container
      const uint32_t tfCounter = static_cast<uint32_t>(currTF);
      for (uint32_t tb = 0; tb < static_cast<uint32_t>(cmvVec.size()); ++tb) {
        mContainer.addEntry(cmvVec[tb], cru, tb, tfCounter);
      }
      ++mProcessedCRUs;
    }

    // Once all CRUs × all TFs have been received, write out
    if (mProcessedCRUs == mCRUs.size() * mTimeFrames) {
      mContainer.nTFs = static_cast<uint32_t>(mTimeFrames);
      mContainer.nCRUs = static_cast<uint32_t>(mCRUs.size());
      LOGP(info, "ProcessedTFs: {}  currTF: {}  relTF: {}  OrbitResetTime: {}  orbits per TF: {}",
           mProcessedCRUs / mCRUs.size(), currTF, relTF, mTFInfo.first, mTFInfo.second);
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "End of stream, flushing CMV container ({} entries, lane {})", mContainer.size(), mLaneId);
    mContainer.nTFs = static_cast<uint32_t>(mTimeFrames);
    mContainer.nCRUs = static_cast<uint32_t>(mCRUs.size());
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
  int mNOrbitsCMV{12};   ///< orbits per CMV integration window (for CCDB timestamp range)
  bool mDumpCMVs{false}; ///< write a local ROOT debug file
  bool mOffsetCCDB{false};
  long mTFFirst{-1};
  long mTimestampStart{0};
  unsigned int mProcessedCRUs{0}; ///< total CRU entries received in this interval
  uint64_t mRun{0};
  dataformats::Pair<long, int> mTFInfo{};
  CMVContainer mContainer{};
  o2::framework::DataTakingContext mDataTakingContext{};
  bool mSetDataTakingCont{true};
  const std::vector<InputSpec> mFilter{
    {"cmvagg",
     ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(mLaneId)},
     Lifetime::Sporadic}};

  /// Determine the CCDB start timestamp from orbit-reset time or framework creation time (depending on mUsePreciseTimestamp).
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

  /// Serialise mContainer into a TMemFile and push to CCDB, then reset state.
  void sendOutput(DataAllocator& output)
  {
    using timer = std::chrono::high_resolution_clock;

    if (mContainer.empty()) {
      LOGP(warning, "CMV container is empty at sendOutput (lane {}), skipping", mLaneId);
      reset();
      return;
    }

    LOGP(info, "{}", mContainer.summary());

    // Compute CCDB validity window
    const long offsetCCDB = mOffsetCCDB ? o2::ccdb::CcdbObjectInfo::HOUR : 0;
    const long timeStampEnd = offsetCCDB + mTimestampStart +
                              mNOrbitsCMV * mTimeFrames * mNTFsBuffer * o2::constants::lhc::LHCOrbitMUS * 0.001;
    LOGP(info, "Setting timestamp range from {} to {} for writing to CCDB with an offset of {}",
         mTimestampStart, timeStampEnd, offsetCCDB);

    if (mSendCCDB && timeStampEnd > mTimestampStart) {
      auto start = timer::now();

      auto tree = mContainer.toTTree();

      // Write local ROOT file for debugging
      if (mDumpCMVs) {
        const std::string fname = fmt::format("CMV_lane{:02}_timestamp{}.root", mLaneId, mTimestampStart);
        try {
          mContainer.writeToFile(fname, tree);
          LOGP(info, "CMV debug file written to {}", fname);
        } catch (const std::exception& e) {
          LOGP(error, "Failed to write CMV debug file: {}", e.what());
        }
      }

      o2::ccdb::CcdbObjectInfo ccdbInfoCMV(
        "TPC/Calib/CMV",
        "TTree",
        "CMV.root",
        /*metadata=*/{},
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
    reset();
  }

  /// Reset all per-interval state
  void reset()
  {
    mContainer.clear();
    mContainer.reserve(mTimeFrames, static_cast<uint32_t>(mCRUs.size()));
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
  inputSpecs.emplace_back(InputSpec{
    "cmvagg",
    ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeCMVSpec::getDataDescriptionCMV(lane)},
    Lifetime::Sporadic});
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
      {"orbits-CMVs", VariantType::Int, 12, {"Number of orbits over which the CMVs are integrated"}},
      {"dump-cmvs", VariantType::Bool, false, {"Dump CMVs to a local ROOT file for debugging"}},
      {"add-offset-for-CCDB-timestamp", VariantType::Bool, false, {"Add an offset of 1 hour for the validity range of the CCDB objects"}}}};

  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif // O2_TPCFACTORIZECMVSPEC_H