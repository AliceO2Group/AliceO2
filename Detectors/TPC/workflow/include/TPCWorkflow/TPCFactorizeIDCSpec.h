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
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned int groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT)
    : mCRUs{crus}, mIDCFactorization{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, timeframes, timeframesDeltaIDC, crus}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug}, mUsePrecisetimeStamp{usePrecisetimeStamp}, mSendOutFFT{sendOutputFFT} {};

  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, TPCFactorizeIDCSpecGroup>::value)), int>::type = 0>
  TPCFactorizeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, std::array<unsigned char, Mapper::NREGIONS> groupPads,
                      std::array<unsigned char, Mapper::NREGIONS> groupRows, std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold,
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned int groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT)
    : mCRUs{crus}, mIDCFactorization{std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, timeframes, timeframesDeltaIDC, crus}, mIDCStruct{TPCFactorizeIDCStruct<TPCFactorizeIDCSpecGroup>(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges)}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug}, mUsePrecisetimeStamp{usePrecisetimeStamp}, mSendOutFFT{sendOutputFFT} {};

  void init(o2::framework::InitContext& ic) final
  {
    mLaneId = ic.services().get<const o2::framework::DeviceSpec>().rank;
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;

    mUpdateGroupingPar = mLaneId == 0 ? !(ic.options().get<bool>("update-not-grouping-parameter")) : false;

    const std::string refGainMapFile = ic.options().get<std::string>("gainMapFile");
    if (!refGainMapFile.empty()) {
      LOGP(info, "Loading GainMap from file {}", refGainMapFile);
      mIDCFactorization.setGainMap(refGainMapFile.data(), "GainMap");
    }

    mTFRangeIDCDelta.resize(mIDCFactorization.getNChunks());
    mTimeStampRangeIDCDelta.resize(mIDCFactorization.getNChunks());
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto currTF = processing_helpers::getCurrentTF(pc);
    // set the min range of TFs for first TF
    if (mProcessedTFs == 0) {
      mTFFirst = currTF;
      mTimeStampCCDB.first = getTimeStamp(pc); // in milliseconds
      LOGP(info, "Setting timestamp validity from for writing to CCDB to {} for TF {}", mTimeStampCCDB.first, currTF);

      // write struct containing grouping parameters to access grouped IDCs to CCDB
      if (mWriteToDB && mUpdateGroupingPar) {
        // validity for grouping parameters is from first TF to some really large TF (until it is updated) TODO do somewhere else?!
        if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
          mDBapi.storeAsTFileAny<o2::tpc::ParameterIDCGroupCCDB>(&mIDCStruct.mIDCs.getIDCGroupHelperSector().getGroupingParameter(), CDBTypeMap.at(CDBType::CalIDCGroupingPar), mMetadata, mTimeStampCCDB.first, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
        } else {
          mDBapi.storeAsTFileAny<o2::tpc::ParameterIDCGroupCCDB>(&mIDCFactorization.getGroupingParameter(), CDBTypeMap.at(CDBType::CalIDCGroupingPar), mMetadata, mTimeStampCCDB.first, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
        }
        mUpdateGroupingPar = false; // write grouping parameters only once
      }

      for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(); ++iChunk) {
        mTFRangeIDCDelta[iChunk] = getFirstTFDeltaIDC(iChunk);
      }
    }

    if (mTFFirst + mIDCFactorization.getNTimeframes() - 1 == currTF) {
      mTimeStampCCDB.second = getTimeStamp(pc); // in milliseconds
      LOGP(info, "Setting timestamp validity to for writing to CCDB to {} for TF {}", mTimeStampCCDB.second, currTF);
    }

    // check if current TF is in range of IDCDelta range
    findTimeStamp(pc);

    for (int i = 0; i < mCRUs.size() + mUsePrecisetimeStamp; ++i) {
      const DataRef ref = pc.inputs().getByPos(i);
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const auto descr = tpcCRUHeader->dataDescription;
      if (TPCDistributeIDCSpec::getDataDescriptionIDC() == descr) {
        const int cru = tpcCRUHeader->subSpecification - mLaneId * CRU::MaxCRU;
        mIDCFactorization.setIDCs(pc.inputs().get<std::vector<float>>(ref), cru, mProcessedTFs); // aggregate IDCs
        LOGP(info, "Received IDCs for CRU {} of size {}", cru, mIDCFactorization.getIDCs()[cru][mProcessedTFs].size());
      }
    }
    ++mProcessedTFs;

    LOGP(info, "aggregated TFs: {}", mProcessedTFs);

    if (mProcessedTFs == mIDCFactorization.getNTimeframes()) {
      mProcessedTFs = 0; // reset processed TFs for next aggregation interval
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

 private:
  const std::vector<uint32_t> mCRUs{};              ///< CRUs to process in this instance
  int mProcessedTFs{0};                             ///< number of processed time frames to keep track of when the writing to CCDB will be done
  IDCFactorization mIDCFactorization;               ///< object aggregating the IDCs and performing the factorization of the IDCs
  TPCFactorizeIDCStruct<Type> mIDCStruct{};         ///< object for averaging and grouping of the IDCs
  const IDCDeltaCompression mCompressionDeltaIDC{}; ///< compression type for IDC Delta
  const bool mDebug{false};                         ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};                  ///< flag if the output will be send (for debugging)
  const bool mUsePrecisetimeStamp{true};            ///< use precise time stamp when writing to CCDB
  const bool mSendOutFFT{false};                    ///<  flag if the output will be send for the FFT
  o2::ccdb::CcdbApi mDBapi;                         ///< API for storing the IDCs in the CCDB
  std::map<std::string, std::string> mMetadata;     ///< meta data of the stored object in CCDB
  bool mWriteToDB{};                                ///< flag if writing to CCDB will be done
  uint32_t mTFFirst{};                              ///< first TF of current aggregation interval
  std::pair<uint64_t, uint64_t> mTimeStampCCDB{};   ///< storing of first and last time stamp range used when setting the validity of the objects when writing to CCDB
  std::vector<uint32_t> mTFRangeIDCDelta{};         ///< tf range for storing IDCDelta
  std::vector<uint64_t> mTimeStampRangeIDCDelta{};  ///< time stamp range of IDCDelta
  bool mUpdateGroupingPar{true};                    ///< flag to set if grouping parameters should be updated or not
  int mLaneId{0};                                   ///< the id of the current process within the parallel pipeline
  std::unique_ptr<CalDet<PadFlags>> mPadFlagsMap;   ///< status flag for each pad (i.e. if the pad is dead). This map is buffered to check if something changed, when a new map is created

  /// \return returns first TF for validity range when storing to CCDB
  uint32_t getFirstTF() const { return mTFFirst; }

  /// \return returns first TF for validity range when storing to IDCDelta CCDB
  unsigned int getFirstTFDeltaIDC(const unsigned int iChunk) const { return getFirstTF() + iChunk * mIDCFactorization.getTimeFramesDeltaIDC(); }

  /// \return returns last TF for validity range when storing to IDCDelta CCDB
  unsigned int getLastTFDeltaIDC(const unsigned int iChunk) const { return (iChunk == mIDCFactorization.getNChunks() - 1) ? (mIDCFactorization.getNTimeframes() + getFirstTF()) : (getFirstTFDeltaIDC(iChunk) + mIDCFactorization.getTimeFramesDeltaIDC()); }

  /// \return returns first time stamp for validity range when storing to IDCDelta CCDB
  auto getFirstTimeStampDeltaIDC(const unsigned int iChunk) const { return mTimeStampRangeIDCDelta[iChunk]; }

  /// \return returns the current timestamp
  auto getTimeStamp(o2::framework::ProcessingContext& pc) const { return mUsePrecisetimeStamp ? processing_helpers::getTimeStamp(pc) / 1000 : processing_helpers::getCreationTime(pc); }

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
      mTimeStampRangeIDCDelta[index] = getTimeStamp(pc);
      // TODO remove found tf?
      return true;
    }
    return false;
  }

  /// send output to next device for debugging
  void sendOutputDebug(DataAllocator& output)
  {
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC0()}, mIDCFactorization.getIDCZero());
    for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(); ++iChunk) {
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDCDelta(), o2::header::DataHeader::SubSpecificationType{iChunk}, Lifetime::Timeframe}, mIDCFactorization.getIDCDeltaUncompressed(iChunk));
    }
  }

  void sendOutput(DataAllocator& output)
  {
    if (mSendOutDebug) {
      sendOutputDebug(output);
    }
    const auto timeStampStart = mTimeStampCCDB.first;
    const auto timeStampEnd = (mTimeStampCCDB.second <= timeStampStart) ? timeStampStart + 1 : mTimeStampCCDB.second;

    // sending output to FFT
    if (mSendOutFFT) {
      LOGP(info, "Sending output");
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::A}}, mIDCFactorization.getIDCOne(Side::A));
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::C}}, mIDCFactorization.getIDCOne(Side::C));
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionTimeStamp()}, std::vector<uint64_t>{timeStampStart, timeStampEnd});
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIntervals()}, mIDCFactorization.getIntegrationIntervalsPerTF());
    }

    if (mWriteToDB) {
      LOGP(info, "Writing IDCs to CCDB");
      mDBapi.storeAsTFileAny<o2::tpc::IDCZero>(&mIDCFactorization.getIDCZero(), CDBTypeMap.at(CDBType::CalIDC0), mMetadata, timeStampStart, timeStampEnd);
      mDBapi.storeAsTFileAny<o2::tpc::IDCOne>(&mIDCFactorization.getIDCOne(), CDBTypeMap.at(CDBType::CalIDC1), mMetadata, timeStampStart, timeStampEnd);

      auto padStatusMap = mIDCFactorization.getPadStatusMap();
      if (padStatusMap) {
        // store map in case it is nullptr
        if (!mPadFlagsMap) {
          mPadFlagsMap = std::move(padStatusMap);
          LOGP(info, "Writing pad status map to CCDB.");
          mDBapi.storeAsTFileAny<CalDet<PadFlags>>(mPadFlagsMap.get(), CDBTypeMap.at(CDBType::CalIDCPadStatusMap), mMetadata, timeStampStart, timeStampEnd);
          LOGP(info, "Pad status map written to CCDB");
        } else {
          // check if map changed. if it changed update the map in the CCDB and store new map in buffer
          if (!(*padStatusMap.get() == *mPadFlagsMap.get())) {
            LOGP(info, "Pad status map changed");
            LOGP(info, "Writing pad status map to CCDB");
            mDBapi.storeAsTFileAny<CalDet<PadFlags>>(mPadFlagsMap.get(), CDBTypeMap.at(CDBType::CalIDCPadStatusMap), mMetadata, timeStampStart, timeStampEnd);
            LOGP(info, "Pad status map written to CCDB");
          }
        }
      }

      for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(); ++iChunk) {
        if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
          // perform grouping of IDC Delta if necessary
          mIDCStruct.mIDCs.setIDCs(std::move(mIDCFactorization).getIDCDeltaUncompressed(iChunk));
          LOGP(info, "averaging and grouping DeltaIDCs for TFs {} - {} for CRUs {} to {} using {} threads", getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk), mCRUs.front(), mCRUs.back(), mIDCStruct.mIDCs.getNThreads());
          mIDCStruct.mIDCs.processIDCs(mPadFlagsMap.get());
          if (mDebug) {
            mIDCStruct.mIDCs.dumpToFile(fmt::format("IDCDeltaAveraged_chunk{:02}_{:02}.root", iChunk, getFirstTFDeltaIDC(iChunk)).data());
          }
        }

        switch (mCompressionDeltaIDC) {
          case IDCDeltaCompression::MEDIUM:
          default: {
            using compType = unsigned short;
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              auto idcDeltaMediumCompressed = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            } else {
              auto idcDeltaMediumCompressed = mIDCFactorization.getIDCDeltaMediumCompressed(iChunk);
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            }

            break;
          }
          case IDCDeltaCompression::HIGH: {
            using compType = unsigned char;
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              auto idcDeltaMediumCompressed = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            } else {
              auto idcDeltaHighCompressed = mIDCFactorization.getIDCDeltaHighCompressed(iChunk);
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaHighCompressed, CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            }
            break;
          }
          case IDCDeltaCompression::NO:
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<float>>(&mIDCStruct.mIDCs.getIDCGroupData(), CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            } else {
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<float>>(&mIDCFactorization.getIDCDeltaUncompressed(iChunk), CDBTypeMap.at(CDBType::CalIDCDelta), mMetadata, getFirstTimeStampDeltaIDC(iChunk), timeStampEnd);
            }
            break;
        }
      }
    }
    // reseting aggregated IDCs. This is done for safety, but if all data is received in the next aggregation interval it isnt necessary... remove it?
    mIDCFactorization.reset();
  }
};

template <class Type>
DataProcessorSpec getTPCFactorizeIDCSpec(const int lane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, const IDCDeltaCompression compression, const bool debug, const bool senddebug, const bool usePrecisetimeStamp, const bool sendOutputFFT)
{
  std::vector<OutputSpec> outputSpecs;
  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC0()});
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDCDelta()});
  }

  if (sendOutputFFT) {
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::A}});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::C}});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionTimeStamp(), header::DataHeader::SubSpecificationType{0}});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIntervals(), header::DataHeader::SubSpecificationType{0}});
  }

  std::vector<InputSpec> inputSpecs;
  inputSpecs.reserve(crus.size());
  for (const auto& cru : crus) {
    inputSpecs.emplace_back(InputSpec{"idcagg", gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(), header::DataHeader::SubSpecificationType{cru + lane * CRU::MaxCRU}, Lifetime::Timeframe});
  }

  if (usePrecisetimeStamp) {
    inputSpecs.emplace_back("orbitreset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset"));
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
    AlgorithmSpec{adaptFromTask<TPCFactorizeIDCSpec<Type>>(crus, timeframes, timeframesDeltaIDC, groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, compression, debug, senddebug, usePrecisetimeStamp, sendOutputFFT)},
    Options{{"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access."}},
            {"gainMapFile", VariantType::String, "", {"file to reference gain map, which will be used for correcting the cluster charge"}},
            {"update-not-grouping-parameter", VariantType::Bool, false, {"Do NOT Update/Writing grouping parameters to CCDB."}}}}; // end DataProcessorSpec
  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif
