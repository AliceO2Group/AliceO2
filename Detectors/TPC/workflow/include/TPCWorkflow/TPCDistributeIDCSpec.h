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
#include "TPCWorkflow/TPCAverageGroupIDCSpec.h"
#include "TPCBase/CRU.h"
#include "MemoryResources/MemoryResources.h"

#include "TFile.h"
#include "TKey.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCDistributeIDCSpec : public o2::framework::Task
{
 public:
  TPCDistributeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const bool loadFromFile, const unsigned int firstTF)
    : mCRUs{crus}, mTimeFrames{timeframes}, mOutLanes{outlanes}, mLoadFromFile{loadFromFile}, mProcessedCRU{{std::vector<unsigned int>(timeframes), std::vector<unsigned int>(timeframes)}}, mTFStart{{firstTF, firstTF + timeframes}}, mTFEnd{{firstTF + timeframes - 1, mTFStart[1] + timeframes - 1}}
  {
    for (auto& idcbuffer : mIDCs) {
      for (auto& idc : idcbuffer) {
        idc.resize(mTimeFrames);
      }
    }
    for (auto& idcbuffer : mOneDIDCs) {
      for (auto& idc : idcbuffer) {
        idc.resize(mTimeFrames);
      }
    }
  };

  void init(o2::framework::InitContext& ic) final
  {
    if (mLoadFromFile) {
      TFile fInp("IDCGroup.root", "READ");
      for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
        const auto key = dynamic_cast<TKey*>(keyAsObj);
        LOGP(info, "Key name: {} Type: {}", key->GetName(), key->GetClassName());

        if (std::strcmp(o2::tpc::IDCAverageGroup::Class()->GetName(), key->GetClassName()) != 0) {
          LOGP(info, "skipping object. wrong class.");
          continue;
        }
        IDCAverageGroup* idcavg = (IDCAverageGroup*)fInp.Get(key->GetName());
        unsigned int cru = idcavg->getSector() * Mapper::NREGIONS + idcavg->getRegion();
        const std::vector<float>& idcData = idcavg->getIDCGroup().getData();
        const std::vector<float>& idc1D = idcavg->getIDCGroup().get1DIDCs();

        for (auto& idcbuffer : mIDCs) {
          const auto nIDCS = idcData.size();
          // TODO use memcyp etc here
          for (auto& relTF : idcbuffer[cru]) {
            relTF.reserve(nIDCS);
            for (int i = 0; i < nIDCS; ++i) {
              relTF.emplace_back(idcData[i]);
            }
          }
        }

        for (auto& idcbuffer : mOneDIDCs) {
          const auto nIDCS = idc1D.size();
          for (auto& relTF : idcbuffer[cru]) {
            relTF.reserve(nIDCS);
            for (int i = 0; i < nIDCS; ++i) {
              relTF.emplace_back(idc1D[i]);
            }
          }
        }
        delete idcavg;
      }
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // check which buffer to use for current incoming data
    const auto tf = getCurrentTF(pc);
    const bool currentBuffer = (tf > mTFEnd[mBuffer]) ? !mBuffer : mBuffer;
    const unsigned int currentOutLane = (tf > mTFEnd[mBuffer]) ? (mCurrentOutLane + 1) % mOutLanes : mCurrentOutLane;
    const auto relTF = tf - mTFStart[currentBuffer];

    if (!mLoadFromFile) {
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
        ++mProcessedCRU[currentBuffer][relTF];
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        const auto descr = tpcCRUHeader->dataDescription;
        if (TPCAverageGroupIDCDevice::getDataDescriptionIDCGroup() == descr) {
          mIDCs[currentBuffer][cru][relTF] = std::move(pc.inputs().get<pmr::vector<float>>(ref));
        } else {
          mOneDIDCs[currentBuffer][cru][relTF] = std::move(pc.inputs().get<pmr::vector<float>>(ref));
        }
      }
    }

    // check if all CRUs for current TF are already aggregated and send data
    if (mProcessedCRU[currentBuffer][relTF] == 2 * mCRUs.size() || mLoadFromFile) {
      ++mProcessedTFs[currentBuffer];
      sendOutput(pc, currentOutLane, currentBuffer, relTF);
    }

    if (mProcessedTFs[currentBuffer] == mTimeFrames) {
      mProcessedTFs[currentBuffer] = 0; // reset processed TFs for next aggregation interval
      std::fill(mProcessedCRU[currentBuffer].begin(), mProcessedCRU[currentBuffer].end(), 0);

      // set integration range for next integration interval
      mTFStart[mBuffer] = mTFEnd[!mBuffer] + 1;
      mTFEnd[mBuffer] = mTFStart[mBuffer] + mTimeFrames - 1;

      // switch buffer
      mBuffer = !mBuffer;

      // set output lane
      mCurrentOutLane = ++mCurrentOutLane % mOutLanes;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  /// return data description for aggregated IDCs
  static constexpr header::DataDescription getDataDescriptionIDC() { return header::DataDescription{"IDCAGG"}; }
  static constexpr header::DataDescription getDataDescription1DIDC() { return header::DataDescription{"1IDCAGG"}; }

 private:
  const std::vector<uint32_t> mCRUs{}; ///< CRUs to process in this instance
  const unsigned int mTimeFrames{};    ///< number of TFs per aggregation interval
  const unsigned int mOutLanes{};      ///< number of output lanes
  const bool mLoadFromFile{};
  std::array<int, 2> mProcessedTFs{{0, 0}};                                            ///< number of processed time frames to keep track of when the writing to CCDB will be done
  std::array<std::array<std::vector<pmr::vector<float>>, CRU::MaxCRU>, 2> mIDCs{};     ///< grouped and integrated IDCs for the whole TPC. CRU -> time frame -> IDCs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::array<std::vector<pmr::vector<float>>, CRU::MaxCRU>, 2> mOneDIDCs{}; ///< 1D IDCs for the whole TPC. CRU -> time frame -> IDCs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::vector<unsigned int>, 2> mProcessedCRU{};                            ///< counter of received data from CRUs per TF to merge incoming data from FLPs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<uint32_t, 2> mTFStart{};                                                  ///< storing of first TF used when setting the validity of the objects when writing to CCDB
  std::array<uint32_t, 2> mTFEnd{};                                                    ///< storing of last TF used when setting the validity of the objects when writing to CCDB
  unsigned int mCurrentOutLane{0};                                                     ///< index for keeping track of the current output lane
  bool mBuffer{false};                                                                 ///< buffer index
  const std::vector<InputSpec> mFilter = {{"idcsgroup", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescriptionIDCGroup()}, Lifetime::Timeframe},
                                          {"1didc", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescription1DIDC()}, Lifetime::Timeframe}}; ///< filter for looping over input data

  /// \return returns TF of current processed data
  uint32_t getCurrentTF(o2::framework::ProcessingContext& pc) const { return o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->tfCounter; }

  void sendOutput(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const bool currentBuffer, const unsigned int relTF)
  {
    // send output data for one TF for all CRUs
    if (!mLoadFromFile) {
      for (unsigned int i = 0; i < mCRUs.size(); ++i) {
        pc.outputs().adoptContainer(Output{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(), header::DataHeader::SubSpecificationType{mCRUs[i] + currentOutLane * CRU::MaxCRU}}, std::move(mIDCs[currentBuffer][mCRUs[i]][relTF]));
        pc.outputs().adoptContainer(Output{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescription1DIDC(), header::DataHeader::SubSpecificationType{mCRUs[i]}}, std::move(mOneDIDCs[currentBuffer][mCRUs[i]][relTF]));
      }
    } else {
      for (unsigned int i = 0; i < mCRUs.size(); ++i) {
        pc.outputs().snapshot(Output{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(), header::DataHeader::SubSpecificationType{mCRUs[i] + currentOutLane * CRU::MaxCRU}}, mIDCs[currentBuffer][mCRUs[i]][relTF]);
        pc.outputs().snapshot(Output{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescription1DIDC(), header::DataHeader::SubSpecificationType{mCRUs[i]}}, mOneDIDCs[currentBuffer][mCRUs[i]][relTF]);
      }
    }
  }
};

DataProcessorSpec getTPCDistributeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const unsigned int firstTF, const bool loadFromFile)
{
  std::vector<InputSpec> inputSpecs;
  if (!loadFromFile) {
    inputSpecs.emplace_back(InputSpec{"idcsgroup", ConcreteDataTypeMatcher{gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescriptionIDCGroup()}, Lifetime::Timeframe});
    inputSpecs.emplace_back(InputSpec{"1didc", ConcreteDataTypeMatcher{gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescription1DIDC()}, Lifetime::Timeframe});
  }

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.reserve((outlanes + 1) * crus.size());
  for (int lane = 0; lane < outlanes; ++lane) {
    for (const auto cru : crus) {
      const header::DataHeader::SubSpecificationType subSpec{cru + lane * CRU::MaxCRU};
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(), subSpec});
    }
  }

  for (const auto cru : crus) {
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescription1DIDC(), header::DataHeader::SubSpecificationType{cru}});
  }

  return DataProcessorSpec{
    "tpc-distribute-idc",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCDistributeIDCSpec>(crus, timeframes, outlanes, loadFromFile, firstTF)}};
}

} // namespace o2::tpc

#endif
