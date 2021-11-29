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
#include <limits>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/IDCFactorization.h"
#include "TPCCalibration/IDCAverageGroup.h"
#include "CCDB/CcdbApi.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/IDCGroupingParameter.h"
#include "TPCWorkflow/TPCDistributeIDCSpec.h"
#include "TPCBase/CRU.h"
#include "CommonUtils/NameConf.h"

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

  TPCFactorizeIDCStruct(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned char groupPadsSectorEdges, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : mIDCs(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges){};
  IDCAverageGroup<IDCAverageGroupTPC> mIDCs; ///< object for averaging and grouping of the IDCs
  inline static int sNThreads{1};            ///< number of threads which are used during the calculations
};

template <class Type>
class TPCFactorizeIDCSpec : public o2::framework::Task
{
 public:
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, TPCFactorizeIDCSpecNoGroup>::value)), int>::type = 0>
  TPCFactorizeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, std::array<unsigned char, Mapper::NREGIONS> groupPads,
                      std::array<unsigned char, Mapper::NREGIONS> groupRows, std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold,
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned char groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug = false, const bool senddebug = false)
    : mCRUs{crus}, mIDCFactorization{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, timeframes, timeframesDeltaIDC}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug} {};

  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, TPCFactorizeIDCSpecGroup>::value)), int>::type = 0>
  TPCFactorizeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, std::array<unsigned char, Mapper::NREGIONS> groupPads,
                      std::array<unsigned char, Mapper::NREGIONS> groupRows, std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold,
                      std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold, const unsigned char groupPadsSectorEdges, const IDCDeltaCompression compression, const bool debug = false, const bool senddebug = false)
    : mCRUs{crus}, mIDCFactorization{std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, timeframes, timeframesDeltaIDC}, mIDCStruct{TPCFactorizeIDCStruct<TPCFactorizeIDCSpecGroup>(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges)}, mCompressionDeltaIDC{compression}, mDebug{debug}, mSendOutDebug{senddebug} {};

  void init(o2::framework::InitContext& ic) final
  {
    mLaneId = ic.services().get<const o2::framework::DeviceSpec>().rank;
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
    mUpdateGroupingPar = mLaneId == 0 ? !(ic.options().get<bool>("update-not-grouping-parameter")) : false;

    // write struct containing grouping parameters to access grouped IDCs to CCDB
    if (mWriteToDB && mUpdateGroupingPar) {
      // validity for grouping parameters is from first TF to some really large TF (until it is updated) TODO do somewhere else?!
      if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
        mDBapi.storeAsTFileAny<o2::tpc::ParameterIDCGroupCCDB>(&mIDCStruct.mIDCs.getIDCGroupHelperSector().getGroupingParameter(), "TPC/Calib/IDC/GROUPINGPAR", mMetadata, getFirstTF(), std::numeric_limits<uint32_t>::max());
      } else {
        mDBapi.storeAsTFileAny<o2::tpc::ParameterIDCGroupCCDB>(&mIDCFactorization.getGroupingParameter(), "TPC/Calib/IDC/GROUPINGPAR", mMetadata, getFirstTF(), std::numeric_limits<uint32_t>::max());
      }
      mUpdateGroupingPar = false; // write grouping parameters only once
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // set the min range of TFs for first TF
    if (mProcessedTFs == 0) {
      mTFRange[0] = getCurrentTF(pc);
    }

    for (int i = 0; i < mCRUs.size(); ++i) {
      const DataRef ref = pc.inputs().getByPos(i);
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification - mLaneId * CRU::MaxCRU;
      mIDCFactorization.setIDCs(pc.inputs().get<std::vector<float>>(ref), cru, mProcessedTFs); // aggregate IDCs
    }
    ++mProcessedTFs;

    if (!(mProcessedTFs % ((mIDCFactorization.getNTimeframes() + 5) / 5))) {
      LOGP(info, "aggregated TFs: {}", mProcessedTFs);
    }

    if (mProcessedTFs == mIDCFactorization.getNTimeframes()) {
      mTFRange[1] = getCurrentTF(pc) + 1; // set the TF for last aggregated TF
      mProcessedTFs = 0;                  // reset processed TFs for next aggregation interval
      if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
        mIDCFactorization.factorizeIDCs(true); // calculate DeltaIDC, 0D-IDC, 1D-IDC
      } else {
        mIDCFactorization.factorizeIDCs(false); // calculate DeltaIDC, 0D-IDC, 1D-IDC
      }

      if (mDebug) {
        LOGP(info, "dumping aggregated and factorized IDCs and FT to file");
        mIDCFactorization.dumpToFile(fmt::format("IDCFactorized_{:02}.root", getCurrentTF(pc)).data());
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
  static constexpr header::DataDescription getDataDescriptionIDCDelta() { return header::DataDescription{"IDCDELTA"}; }

 private:
  const std::vector<uint32_t> mCRUs{};              ///< CRUs to process in this instance
  int mProcessedTFs{0};                             ///< number of processed time frames to keep track of when the writing to CCDB will be done
  IDCFactorization mIDCFactorization;               ///< object aggregating the IDCs and performing the factorization of the IDCs
  TPCFactorizeIDCStruct<Type> mIDCStruct{};         ///< object for averaging and grouping of the IDCs
  const IDCDeltaCompression mCompressionDeltaIDC{}; ///< compression type for IDC Delta
  const bool mDebug{false};                         ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};                  ///< flag if the output will be send (for debugging)
  o2::ccdb::CcdbApi mDBapi;                         ///< API for storing the IDCs in the CCDB
  std::map<std::string, std::string> mMetadata;     ///< meta data of the stored object in CCDB
  bool mWriteToDB{};                                ///< flag if writing to CCDB will be done
  std::array<uint32_t, 2> mTFRange{};               ///< storing of first and last TF used when setting the validity of the objects when writing to CCDB
  bool mUpdateGroupingPar{true};                    ///< flag to set if grouping parameters should be updated or not
  int mLaneId{0};                                   ///< the id of the current process within the parallel pipeline

  /// \return returns TF of current processed data
  uint32_t getCurrentTF(o2::framework::ProcessingContext& pc) const { return o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->tfCounter; }

  /// \return returns first TF for validity range when storing to CCDB
  uint32_t getFirstTF() const { return mTFRange[0]; }

  /// \return returns last TF for validity range when storing to CCDB
  uint32_t getLastTF() const { return mTFRange[1]; }

  /// \return returns first TF for validity range when storing to IDCDelta CCDB
  unsigned int getFirstTFDeltaIDC(const unsigned int iChunk) const { return getFirstTF() + iChunk * mIDCFactorization.getTimeFramesDeltaIDC(); }

  /// \return returns last TF for validity range when storing to IDCDelta CCDB
  unsigned int getLastTFDeltaIDC(const unsigned int iChunk) const { return (iChunk == mIDCFactorization.getNChunks() - 1) ? (mIDCFactorization.getNTimeframes() + getFirstTF()) : (getFirstTFDeltaIDC(iChunk) + mIDCFactorization.getTimeFramesDeltaIDC()); }

  /// send output to next device for debugging
  void sendOutputDebug(DataAllocator& output)
  {
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC0()}, mIDCFactorization.getIDCZero());
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDC1()}, mIDCFactorization.getIDCOne());
    for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(); ++iChunk) {
      output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDCDelta(), o2::header::DataHeader::SubSpecificationType{iChunk}, Lifetime::Timeframe}, mIDCFactorization.getIDCDeltaUncompressed(iChunk));
    }
  }

  void sendOutput(DataAllocator& output)
  {
    if (mSendOutDebug) {
      sendOutputDebug(output);
    }

    if (mWriteToDB) {
      const long timeStampStart = getFirstTF();
      const long timeStampEnd = getLastTF();
      mDBapi.storeAsTFileAny<o2::tpc::IDCZero>(&mIDCFactorization.getIDCZero(), "TPC/Calib/IDC/IDC0", mMetadata, timeStampStart, timeStampEnd);
      mDBapi.storeAsTFileAny<o2::tpc::IDCOne>(&mIDCFactorization.getIDCOne(), "TPC/Calib/IDC/IDC1", mMetadata, timeStampStart, timeStampEnd);

      for (unsigned int iChunk = 0; iChunk < mIDCFactorization.getNChunks(); ++iChunk) {
        if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
          mIDCStruct.mIDCs.setIDCs(std::move(mIDCFactorization).getIDCDeltaUncompressed(iChunk));
          LOGP(info, "averaging and grouping DeltaIDCs for TFs {} - {} for CRUs {} to {} using {} threads", getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk), mCRUs.front(), mCRUs.back(), mIDCStruct.mIDCs.getNThreads());
          mIDCStruct.mIDCs.processIDCs();
          if (mDebug) {
            mIDCStruct.mIDCs.dumpToFile(fmt::format("IDCDeltaAveraged_chunk{:02}_{:02}.root", iChunk, getFirstTFDeltaIDC(iChunk)).data());
          }
        }

        switch (mCompressionDeltaIDC) {
          case IDCDeltaCompression::MEDIUM:
          default: {
            using compType = short;
            // perform grouping of IDC Delta if necessary
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              auto idcDeltaMediumCompressed = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
            } else {
              auto idcDeltaMediumCompressed = mIDCFactorization.getIDCDeltaMediumCompressed(iChunk);
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
            }

            break;
          }
          case IDCDeltaCompression::HIGH: {
            using compType = char;
            // perform grouping of IDC Delta if necessary
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              auto idcDeltaMediumCompressed = IDCDeltaCompressionHelper<compType>::getCompressedIDCs(mIDCStruct.mIDCs.getIDCGroupData());
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaMediumCompressed, "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
            } else {
              auto idcDeltaHighCompressed = mIDCFactorization.getIDCDeltaHighCompressed(iChunk);
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<compType>>(&idcDeltaHighCompressed, "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
            }
            break;
          }
          case IDCDeltaCompression::NO:
            if constexpr (std::is_same_v<Type, TPCFactorizeIDCSpecGroup>) {
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<float>>(&mIDCStruct.mIDCs.getIDCGroupData(), "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
            } else {
              mDBapi.storeAsTFileAny<o2::tpc::IDCDelta<float>>(&mIDCFactorization.getIDCDeltaUncompressed(iChunk), "TPC/Calib/IDC/IDCDELTA", mMetadata, getFirstTFDeltaIDC(iChunk), getLastTFDeltaIDC(iChunk));
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
DataProcessorSpec getTPCFactorizeIDCSpec(const int lane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int timeframesDeltaIDC, const IDCDeltaCompression compression, const bool debug = false, const bool senddebug = false)
{
  std::vector<OutputSpec> outputSpecs;
  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC0()});
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDC1()});
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<Type>::getDataDescriptionIDCDelta()});
  }

  std::vector<InputSpec> inputSpecs; //{InputSpec{"idcagg", ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC()}, Lifetime::Timeframe}};
  inputSpecs.reserve(crus.size());
  for (const auto& cru : crus) {
    inputSpecs.emplace_back(InputSpec{"idcagg", gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(), header::DataHeader::SubSpecificationType{cru + lane * CRU::MaxCRU}, Lifetime::Timeframe});
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
  const unsigned char groupPadsSectorEdges = paramIDCGroup.groupPadsSectorEdges;

  DataProcessorSpec spec{
    fmt::format("tpc-factorize-idc-{:02}", lane).data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFactorizeIDCSpec<Type>>(crus, timeframes, timeframesDeltaIDC, groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, compression, debug, senddebug)},
    Options{{"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access."}},
            {"update-not-grouping-parameter", VariantType::Bool, false, {"Do NOT Update/Writing grouping parameters to CCDB."}}}}; // end DataProcessorSpec
  spec.rank = lane;
  return spec;
}

} // namespace o2::tpc

#endif
