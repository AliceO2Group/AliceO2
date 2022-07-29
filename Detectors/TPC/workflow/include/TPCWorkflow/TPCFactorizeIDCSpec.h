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

/// \file TPCFactorizeGroupedIDCSpec.h
/// \brief TPC aggregation of grouped IDCs and factorization
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jun 25, 2021

#ifndef O2_TPCFACTORIZEIDCSPEC_H
#define O2_TPCFACTORIZEIDCSPEC_H

#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/IDCFactorization.h"
#include "TPCCalibration/IDCAverageGroup.h"
#include "CCDB/CcdbApi.h"
#include "TPCWorkflow/TPCDistributeIDCSpec.h"
#include "TPCBase/CRU.h"
#include "CommonUtils/NameConf.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/CDBInterface.h"
#include "DetectorsCalibration/Utils.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

template <class Type>
struct TPCFactorizeIDCStruct;

/// dummy class for template specialization
class TPCFactorizeIDCSpecGroup;
class TPCFactorizeIDCSpecNoGroup;

template <>
struct TPCFactorizeIDCStruct<TPCFactorizeIDCSpecNoGroup> {
};

template <>
struct TPCFactorizeIDCStruct<TPCFactorizeIDCSpecGroup> {

  TPCFactorizeIDCStruct(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupPadsSectorEdges, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : mIDCs(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges){};
  IDCAverageGroup<IDCAverageGroupTPC> mIDCs; ///< object for averaging and grouping of the IDCs
  inline static int sNThreads{1};            ///< number of threads which are used during the calculations
};

template <class Type = TPCFactorizeIDCSpecNoGroup>
class TPCFactorizeIDCSpec : public o2::framework::Task
{
 public:
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, TPCFactorizeIDCSpecNoGroup>::value)), int>::type = 0>
  TPCFactorizeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, std::array<unsigned char, Mapper::NREGIONS> groupPads,
                      std::array<unsigned char, Mapper::NREGIONS> groupRows, std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold,
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned int groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT, const bool sendCCDB, const int lane, const std::vector<o2::tpc::Side>& sides)
    : mCRUs{crus}, mIDCFactorization{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, timeframes, timeframesDeltaIDC, crus}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug}, mUsePrecisetimeStamp{usePrecisetimeStamp}, mSendOutFFT{sendOutputFFT}, mSendOutCCDB{sendCCDB}, mLaneId{lane}, mSides{sides} {};

  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, TPCFactorizeIDCSpecGroup>::value)), int>::type = 0>
  TPCFactorizeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, std::array<unsigned char, Mapper::NREGIONS> groupPads,
                      std::array<unsigned char, Mapper::NREGIONS> groupRows, std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold,
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned int groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT, const bool sendCCDB, const int lane, const std::vector<o2::tpc::Side>& sides)
    : mCRUs{crus}, mIDCFactorization{std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, timeframes, timeframesDeltaIDC, crus}, mIDCStruct{TPCFactorizeIDCStruct<TPCFactorizeIDCSpecGroup>(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges)}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug}, mUsePrecisetimeStamp{usePrecisetimeStamp}, mSendOutFFT{sendOutputFFT}, mSendOutCCDB{sendCCDB}, mLaneId{lane}, mSides{sides} {};

  void init(o2::framework::InitContext& ic) final
  {
    mUpdateGroupingPar = mLaneId == 0 ? !(ic.options().get<bool>("update-not-grouping-parameter")) : false;
    mIDCFactorization.setUsePadStatusMap(ic.options().get<bool>("enablePadStatusMap"));
    mDisableWritingPadStatusMap = ic.options().get<bool>("disableWritingPadStatusMap");
    mLanesDistribute = ic.options().get<int>("lanesDistribute");
    mTFsMessaged = ic.options().get<int>("nTFsMessage") * mCRUs.size();

    const std::string refGainMapFile = ic.options().get<std::string>("gainMapFile");
    if (!refGainMapFile.empty()) {
      LOGP(info, "Loading GainMap from file {}", refGainMapFile);
      mIDCFactorization.setGainMap(refGainMapFile.data(), "GainMap");
    }

    mTFRangeIDCDelta.resize(mIDCFactorization.getNChunks(mSides.front()));
    mTimeStampRangeIDCDelta.resize(mIDCFactorization.getNChunks(mSides.front()));
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // store precise timestamp for look up later
    if (mUsePrecisetimeStamp && pc.inputs().isValid("orbitreset")) {
      mOrbitResetTime = pc.inputs().get<Long64_t>("orbitreset");
      if (pc.inputs().countValidInputs() == 1) {
        return;
      }
    }

    const auto currTF = processing_helpers::getCurrentTF(pc);

    // set the min range of TFs for first TF
    if (mProcessedCRUs == 0) {
      mTFOrbitFirst = processing_helpers::getFirstTForbit(pc);
      mTFFirst = currTF;
      for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(mSides.front()); ++iChunk) {
        mTFRangeIDCDelta[iChunk] = getFirstTFDeltaIDC(iChunk);
      }
    }

    // relative TF of current aggreagtion intervall
    const auto relTF = currTF - mTFFirst;

    // check if current TF is in range of IDCDelta range
    findTimeStamp(pc);

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification;
      mIDCFactorization.setIDCs(pc.inputs().get<std::vector<float>>(ref), cru, relTF); // aggregate IDCs
      ++mProcessedCRUs;
    }

    if (!(mProcessedCRUs % mTFsMessaged)) {
      LOGP(info, "ProcessedTFs: {}   currTF: {}  relTF: {}   OrbitResetTime: {}", mProcessedCRUs / mCRUs.size(), currTF, relTF, mOrbitResetTime);
    }

    if (mProcessedCRUs == mCRUs.size() * mIDCFactorization.getNTimeframes()) {
      mProcessedCRUs = 0; // reset processed TFs for next aggregation interval
      if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
        mIDCFactorization.factorizeIDCs(true); // calculate DeltaIDC, 0D-IDC, 1D-IDC
      } else {
        mIDCFactorization.factorizeIDCs(false); // calculate DeltaIDC, 0D-IDC, 1D-IDC
      }

      if (mDebug) {
        LOGP(info, "dumping aggregated and factorized IDCs to file");
        mIDCFactorization.dumpToFile(fmt::format("IDCFactorized_{:02}.root", currTF).data());
        mIDCFactorization.dumpPadFlagMap("padstatusmap.root", "PadStatus");
      }

      // storing to CCDB
      sendOutput(pc.outputs());
      mTFOrbitFirst = 0;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionIDC0() { return header::DataDescription{"IDC0"}; }
  static constexpr header::DataDescription getDataDescriptionIDC1() { return header::DataDescription{"IDC1"}; }
  static constexpr header::DataDescription getDataDescriptionTimeStamp() { return header::DataDescription{"FOURIERTS"}; }
  static constexpr header::DataDescription getDataDescriptionIntervals() { return header::DataDescription{"INTERVALS"}; }
  static constexpr header::DataDescription getDataDescriptionIDCDelta() { return header::DataDescription{"IDCDELTA"}; }
  static constexpr header::DataDescription getDataDescriptionFourier() { return header::DataDescription{"FOURIER"}; }
  static constexpr header::DataDescription getDataDescriptionLane() { return header::DataDescription{"IDCLANE"}; }

  // for CCDB
  static constexpr header::DataDescription getDataDescriptionCCDBGroupingPar() { return header::DataDescription{"TPC_CalibGrParam"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBIDC0() { return header::DataDescription{"TPC_CalibIDC0"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBIDC1() { return header::DataDescription{"TPC_CalibIDC1"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBIDCDelta() { return header::DataDescription{"TPC_IDCDelta"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBIDCPadFlag() { return header::DataDescription{"TPC_CalibFlags"}; }

 private:
  const std::vector<uint32_t> mCRUs{};                                                                                                                                     ///< CRUs to process in this instance
  int mProcessedCRUs{};                                                                                                                                                    ///< number of processed CRUs to keep track of when the writing to CCDB etc. will be done
  IDCFactorization mIDCFactorization;                                                                                                                                      ///< object aggregating the IDCs and performing the factorization of the IDCs
  TPCFactorizeIDCStruct<Type> mIDCStruct{};                                                                                                                                ///< object for averaging and grouping of the IDCs
  const IDCDeltaCompression mCompressionDeltaIDC{};                                                                                                                        ///< compression type for IDC Delta
  const bool mDebug{false};                                                                                                                                                ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};                                                                                                                                         ///< flag if the output will be send (for debugging)
  const bool mUsePrecisetimeStamp{true};                                                                                                                                   ///< use precise time stamp when writing to CCDB
  const bool mSendOutFFT{false};                                                                                                                                           ///<  flag if the output will be send for the FFT
  const bool mSendOutCCDB{false};                                                                                                                                          ///< sending the outputs for ccdb populator
  uint32_t mTFFirst{};                                                                                                                                                     ///< first TF of current aggregation interval
  std::vector<uint32_t> mTFRangeIDCDelta{};                                                                                                                                ///< tf range for storing IDCDelta
  std::vector<uint64_t> mTimeStampRangeIDCDelta{};                                                                                                                         ///< time stamp range of IDCDelta
  bool mUpdateGroupingPar{true};                                                                                                                                           ///< flag to set if grouping parameters should be updated or not
  const int mLaneId{0};                                                                                                                                                    ///< the id of the current process within the parallel pipeline
  std::vector<Side> mSides{};                                                                                                                                              ///< processed TPC sides
  std::unique_ptr<CalDet<PadFlags>> mPadFlagsMap;                                                                                                                          ///< status flag for each pad (i.e. if the pad is dead). This map is buffered to check if something changed, when a new map is created
  int mLanesDistribute{1};                                                                                                                                                 ///< number of lanes used in the DistributeIDC device
  unsigned int mTFsMessaged{10};                                                                                                                                           ///< send info messages only every mTFsMessaged
  Long64_t mOrbitResetTime{};                                                                                                                                              ///< orbit reset time for CCDB time stamp writing
  uint32_t mTFOrbitFirst{};                                                                                                                                                ///< first TF orbit of current aggregation interval
  bool mDisableWritingPadStatusMap{false};                                                                                                                                 ///< do not store the pad status map in the CCDB
  const std::vector<InputSpec> mFilter = {{"idcagg", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(mLaneId)}, Lifetime::Timeframe}}; ///< filter for looping over input data

  /// \return returns first TF for validity range when storing to CCDB
  uint32_t getFirstTF() const { return mTFFirst; }

  /// \return returns first TF for validity range when storing to IDCDelta CCDB
  unsigned int getFirstTFDeltaIDC(const unsigned int iChunk) const { return getFirstTF() + iChunk * mIDCFactorization.getTimeFramesDeltaIDC(); }

  /// \return returns last TF for validity range when storing to IDCDelta CCDB
  unsigned int getLastTFDeltaIDC(const unsigned int iChunk) const { return (iChunk == mIDCFactorization.getNChunks(mSides.front()) - 1) ? (mIDCFactorization.getNTimeframes() + getFirstTF()) : (getFirstTFDeltaIDC(iChunk) + mIDCFactorization.getTimeFramesDeltaIDC()); }

  /// \return returns first time stamp for validity range when storing to IDCDelta CCDB
  auto getFirstTimeStampDeltaIDC(const unsigned int iChunk) const { return mTimeStampRangeIDCDelta[iChunk]; }

  /// \return returns the current timestamp
  auto getTimeStamp(o2::framework::ProcessingContext& pc) { return mUsePrecisetimeStamp ? processing_helpers::getTimeStamp(pc, mOrbitResetTime) / 1000 : processing_helpers::getCreationTime(pc); }

  /// check if current tf will be used to set the time stamp range
  bool findTimeStamp(o2::framework::ProcessingContext& pc)
  {
    const auto tf = processing_helpers::getCurrentTF(pc);
    return findTimeStamp(tf, pc);
  }

  bool findTimeStamp(const uint32_t tf, o2::framework::ProcessingContext& pc)
  {
    auto it = std::find(mTFRangeIDCDelta.begin(), mTFRangeIDCDelta.end(), tf);
    if (it != mTFRangeIDCDelta.end()) {
      const int index = std::distance(mTFRangeIDCDelta.begin(), it);
      mTimeStampRangeIDCDelta[index] = mUsePrecisetimeStamp ? processing_helpers::getFirstTForbit(pc) : getTimeStamp(pc);
      // TODO remove found tf?
      return true;
    }
    return false;
  }

  /// send output to next device for debugging
  void sendOutputDebug(DataAllocator& output)
  {
    for (const auto side : mSides) {
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC0()}, mIDCFactorization.getIDCZero(side));
      for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(side); ++iChunk) {
        output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDCDelta(), o2::header::DataHeader::SubSpecificationType{iChunk}, Lifetime::Timeframe}, mIDCFactorization.getIDCDeltaUncompressed(iChunk, side));
      }
    }
  }

  void sendOutput(DataAllocator& output)
  {
    using timer = std::chrono::high_resolution_clock;

    if (mSendOutDebug) {
      sendOutputDebug(output);
    }

    const auto timeStampStart = mUsePrecisetimeStamp ? (processing_helpers::getTimeStamp(mOrbitResetTime, mTFOrbitFirst) / 1000) : getFirstTimeStampDeltaIDC(0);
    const auto timeStampEnd = timeStampStart + o2::ccdb::CcdbObjectInfo::DAY;

    // sending output to FFT
    if (mSendOutFFT) {
      for (const auto side : mSides) {
        const unsigned int iSide = static_cast<int>(side);
        LOGP(info, "Sending IDC1 for side {} of size {}", iSide, mIDCFactorization.getIDCOneVec(side).size());
        output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{iSide}}, mIDCFactorization.getIDCOneVec(side));
      }
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionTimeStamp()}, std::vector<uint64_t>{timeStampStart, timeStampEnd});
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIntervals()}, mIDCFactorization.getIntegrationIntervalsPerTF());
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionLane()}, mLaneId);
    }

    if (mSendOutCCDB) {
      for (int iSide = 0; iSide < mSides.size(); ++iSide) {
        const Side side = mSides[iSide];
        LOGP(info, "Writing IDCs to CCDB for Side {}", static_cast<int>(side));
        const bool sideA = side == Side::A;

        // write struct containing grouping parameters to access grouped IDCs to CCDB
        if (mUpdateGroupingPar) {
          ParameterIDCGroupCCDB object;
          if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
            object = mIDCStruct.mIDCs.getIDCGroupHelperSector().getGroupingParameter();
          } else {
            object = mIDCFactorization.getGroupingParameter();
          }

          o2::ccdb::CcdbObjectInfo ccdbInfo(CDBTypeMap.at(sideA ? CDBType::CalIDCGroupingParA : CDBType::CalIDCGroupingParC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
          auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &ccdbInfo);
          LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfo.getPath(), ccdbInfo.getFileName(), image->size(), ccdbInfo.getStartValidityTimestamp(), ccdbInfo.getEndValidityTimestamp());
          output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBGroupingPar(), 0}, *image.get());
          output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBGroupingPar(), 0}, ccdbInfo);
          mUpdateGroupingPar = false; // write grouping parameters only once
        }

        auto start = timer::now();
        o2::ccdb::CcdbObjectInfo ccdbInfoIDC0(CDBTypeMap.at(sideA ? CDBType::CalIDC0A : CDBType::CalIDC0C), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, timeStampEnd);
        auto imageIDC0 = o2::ccdb::CcdbApi::createObjectImage(&mIDCFactorization.getIDCZero(side), &ccdbInfoIDC0);
        LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoIDC0.getPath(), ccdbInfoIDC0.getFileName(), imageIDC0->size(), ccdbInfoIDC0.getStartValidityTimestamp(), ccdbInfoIDC0.getEndValidityTimestamp());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBIDC0(), 0}, *imageIDC0.get());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBIDC0(), 0}, ccdbInfoIDC0);
        auto stop = timer::now();
        std::chrono::duration<float> time = stop - start;
        float totalTime = time.count();
        LOGP(info, "IDCZero CCDB time: {}", time.count());

        start = timer::now();
        o2::ccdb::CcdbObjectInfo ccdbInfoIDC1(CDBTypeMap.at(sideA ? CDBType::CalIDC1A : CDBType::CalIDC1C), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, timeStampEnd);
        auto imageIDC1 = o2::ccdb::CcdbApi::createObjectImage(&mIDCFactorization.getIDCOne(side), &ccdbInfoIDC1);
        LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoIDC1.getPath(), ccdbInfoIDC1.getFileName(), imageIDC1->size(), ccdbInfoIDC1.getStartValidityTimestamp(), ccdbInfoIDC1.getEndValidityTimestamp());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBIDC1(), 0}, *imageIDC1.get());
        output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBIDC1(), 0}, ccdbInfoIDC1);
        stop = timer::now();
        time = stop - start;
        LOGP(info, "IDC1 CCDB time: {}", time.count());
        totalTime += time.count();

        auto padStatusMap = mIDCFactorization.getPadStatusMap();
        if (padStatusMap && iSide == 0) {
          start = timer::now();

          // store map in case it is nullptr
          if (!mDisableWritingPadStatusMap) {
            if (!mPadFlagsMap) {
              mPadFlagsMap = std::move(padStatusMap);
              LOGP(info, "Writing pad status map to CCDB.");
              o2::ccdb::CcdbObjectInfo ccdbInfoPadFlags(CDBTypeMap.at(sideA ? CDBType::CalIDCPadStatusMapA : CDBType::CalIDCPadStatusMapC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, timeStampEnd);
              auto imageFlagMap = o2::ccdb::CcdbApi::createObjectImage(mPadFlagsMap.get(), &ccdbInfoPadFlags);
              LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoPadFlags.getPath(), ccdbInfoPadFlags.getFileName(), imageFlagMap->size(), ccdbInfoPadFlags.getStartValidityTimestamp(), ccdbInfoPadFlags.getEndValidityTimestamp());
              output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBIDCPadFlag(), 0}, *imageFlagMap.get());
              output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBIDCPadFlag(), 0}, ccdbInfoPadFlags);
              LOGP(info, "Pad status map written to CCDB");
            } else {
              // check if map changed. if it changed update the map in the CCDB and store new map in buffer
              if (!(*padStatusMap.get() == *mPadFlagsMap.get())) {
                LOGP(info, "Pad status map changed");
                LOGP(info, "Writing pad status map to CCDB");
                o2::ccdb::CcdbObjectInfo ccdbInfoPadFlags(CDBTypeMap.at(sideA ? CDBType::CalIDCPadStatusMapA : CDBType::CalIDCPadStatusMapC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStart, timeStampEnd);
                auto imageFlagMap = o2::ccdb::CcdbApi::createObjectImage(mPadFlagsMap.get(), &ccdbInfoPadFlags);
                LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoPadFlags.getPath(), ccdbInfoPadFlags.getFileName(), imageFlagMap->size(), ccdbInfoPadFlags.getStartValidityTimestamp(), ccdbInfoPadFlags.getEndValidityTimestamp());
                output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBIDCPadFlag(), 0}, *imageFlagMap.get());
                output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBIDCPadFlag(), 0}, ccdbInfoPadFlags);
                LOGP(info, "Pad status map written to CCDB");
              }
              stop = timer::now();
              time = stop - start;
              LOGP(info, "Pad status map CCDB time: {}", time.count());
              totalTime += time.count();
            }
          }
          stop = timer::now();
          time = stop - start;
          LOGP(info, "Pad status map CCDB time: {}", time.count());
          totalTime += time.count();
        }

        start = timer::now();
        for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(side); ++iChunk) {
          if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
            // perform grouping of IDC Delta if necessary
            auto startGrouping = timer::now();
            mIDCStruct.mIDCs.setIDCs(std::move(mIDCFactorization).getIDCDeltaUncompressed(iChunk, side), side);
            LOGP(info, "averaging and grouping DeltaIDCs for TFs {} - {} for CRUs {} to {} using {} threads", getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk), mCRUs.front(), mCRUs.back(), mIDCStruct.mIDCs.getNThreads());

            // perform averagiing and grouping
            mIDCStruct.mIDCs.processIDCs(mIDCFactorization.getUsePadStatusMap() ? mPadFlagsMap.get() : nullptr);

            auto stopGrouping = timer::now();
            time = stopGrouping - startGrouping;
            LOGP(info, "Averaging and grouping time: {}", time.count());

            if (mDebug) {
              mIDCStruct.mIDCs.dumpToFile(fmt::format("IDCDeltaAveraged_chunk{:02}_{:02}_side{}.root", iChunk, getFirstTFDeltaIDC(iChunk), side).data());
            }
          }

          const auto timeStampStartDelta = mUsePrecisetimeStamp ? (processing_helpers::getTimeStamp(mOrbitResetTime, getFirstTimeStampDeltaIDC(iChunk)) / 1000) : getFirstTimeStampDeltaIDC(iChunk);
          o2::ccdb::CcdbObjectInfo ccdbInfoIDCDelta(CDBTypeMap.at(sideA ? CDBType::CalIDCDeltaA : CDBType::CalIDCDeltaC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampStartDelta, timeStampEnd);

          auto startCCDBIDCDelta = timer::now();
          std::unique_ptr<std::vector<char>> imageIDCDelta;

          switch (mCompressionDeltaIDC) {
            case IDCDeltaCompression::MEDIUM:
            default: {
              using compType = unsigned short;
              IDCDelta<compType> idcDelta;
              if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
                idcDelta = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              } else {
                idcDelta = mIDCFactorization.getIDCDeltaMediumCompressed(iChunk, side);
              }
              imageIDCDelta = o2::ccdb::CcdbApi::createObjectImage(&idcDelta, &ccdbInfoIDCDelta);
              break;
            }
            case IDCDeltaCompression::HIGH: {
              using compType = unsigned char;
              IDCDelta<compType> idcDelta;
              if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
                idcDelta = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              } else {
                idcDelta = mIDCFactorization.getIDCDeltaHighCompressed(iChunk, side);
              }
              imageIDCDelta = o2::ccdb::CcdbApi::createObjectImage(&idcDelta, &ccdbInfoIDCDelta);
              break;
            }
            case IDCDeltaCompression::NO:
              IDCDelta<float> idcDelta;
              if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
                idcDelta = std::move(mIDCStruct).mIDCs.getIDCGroupData();
              } else {
                idcDelta = std::move(mIDCFactorization).getIDCDeltaUncompressed(iChunk, side);
              }
              imageIDCDelta = o2::ccdb::CcdbApi::createObjectImage(&idcDelta, &ccdbInfoIDCDelta);
              break;
          }
          LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoIDCDelta.getPath(), ccdbInfoIDCDelta.getFileName(), imageIDCDelta->size(), ccdbInfoIDCDelta.getStartValidityTimestamp(), ccdbInfoIDCDelta.getEndValidityTimestamp());
          output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBIDCDelta(), iChunk}, *imageIDCDelta.get());
          output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBIDCDelta(), iChunk}, ccdbInfoIDCDelta);
          auto stopCCDBIDCDelta = timer::now();
          time = stopCCDBIDCDelta - startCCDBIDCDelta;
          LOGP(info, "Compression and CCDB object creation time: {}", time.count());
        }

        stop = timer::now();
        time = stop - start;
        LOGP(info, "IDCDelta CCDB time: {}", time.count());
        totalTime += time.count();
        LOGP(info, "CCDB object creation done. Total time: {}", totalTime);
      }
    }

    // reseting aggregated IDCs. This is done for safety, but if all data is received in the next aggregation interval it isnt necessary... remove it?
    LOGP(info, "Everything done! Clearing memory...");
    mIDCFactorization.reset();
    LOGP(info, "Everything cleared. Waiting for new data to arrive.");
  }
};

template <class Type>
DataProcessorSpec getTPCFactorizeIDCSpec(const int lane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT, const bool sendCCDB)
{
  const auto sides = o2::tpc::IDCFactorization::getSides(crus);

  std::vector<OutputSpec> outputSpecs;
  if (sendCCDB) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBGroupingPar()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBGroupingPar()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDC0()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDC0()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDC1()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDC1()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDCDelta()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDCDelta()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDCPadFlag()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFactorizeIDCSpec<Type>::getDataDescriptionCCDBIDCPadFlag()}, Lifetime::Sporadic);
  }

  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC0()}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDCDelta()}, Lifetime::Sporadic);
  }

  if (sendOutputFFT) {
    for (auto side : sides) {
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{side}}, Lifetime::Sporadic);
    }
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionTimeStamp(), header::DataHeader::SubSpecificationType{0}}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIntervals(), header::DataHeader::SubSpecificationType{0}}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionLane(), header::DataHeader::SubSpecificationType{0}}, Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"idcagg", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(lane)}, Lifetime::Sporadic});

  if (usePrecisetimeStamp) {
    inputSpecs.emplace_back(InputSpec{"orbitreset", gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDCOrbitReset(), header::DataHeader::SubSpecificationType{static_cast<unsigned int>(lane)}, Lifetime::Sporadic});
  }

  const auto& paramIDCGroup = ParameterIDCGroup::Instance();
  std::array<unsigned char, Mapper::NREGIONS> groupPads{};
  std::array<unsigned char, Mapper::NREGIONS> groupRows{};
  std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold{};
  std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold{};
  std::copy(std::begin(paramIDCGroup.groupPads), std::end(paramIDCGroup.groupPads), std::begin(groupPads));
  std::copy(std::begin(paramIDCGroup.groupRows), std::end(paramIDCGroup.groupRows), std::begin(groupRows));
  std::copy(std::begin(paramIDCGroup.groupLastRowsThreshold), std::end(paramIDCGroup.groupLastRowsThreshold), std::begin(groupLastRowsThreshold));
  std::copy(std::begin(paramIDCGroup.groupLastPadsThreshold), std::end(paramIDCGroup.groupLastPadsThreshold), std::begin(groupLastPadsThreshold));
  const unsigned int groupPadsSectorEdges = paramIDCGroup.groupPadsSectorEdges;

  DataProcessorSpec spec{
    fmt::format("tpc-factorize-idc-{:02}", lane).data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFactorizeIDCSpec<Type>>(crus, timeframes, timeframesDeltaIDC, groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, compression, debug, senddebug, usePrecisetimeStamp, sendOutputFFT, sendCCDB, lane, sides)},
    Options{{"gainMapFile", VariantType::String, "", {"file to reference gain map, which will be used for correcting the cluster charge"}},
            {"lanesDistribute", VariantType::Int, 1, {"Number of lanes which were used in the DistributeIDC device."}},
            {"nTFsMessage", VariantType::Int, 200, {"Send messages only every nTFs."}},
            {"enablePadStatusMap", VariantType::Bool, false, {"Enabling the usage of the pad-by-pad status map during factorization."}},
            {"disableWritingPadStatusMap", VariantType::Bool, false, {"Do not write the pad status map to CCDB."}},
            {"update-not-grouping-parameter", VariantType::Bool, false, {"Do NOT Update/Writing grouping parameters to CCDB."}}}}; // end DataProcessorSpec
  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif
