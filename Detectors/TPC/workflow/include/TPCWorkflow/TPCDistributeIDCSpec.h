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
#include "TPCWorkflow/TPCFLPIDCSpec.h"
#include "TPCBase/CRU.h"
#include "MemoryResources/MemoryResources.h"
#include "TPCWorkflow/ProcessingHelpers.h"

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
  TPCDistributeIDCSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const bool loadFromFile, const int firstTF, const bool sendPrecisetimeStamp)
    : mCRUs{crus}, mTimeFrames{timeframes}, mOutLanes{outlanes}, mLoadFromFile{loadFromFile}, mProcessedCRU{{std::vector<unsigned int>(timeframes), std::vector<unsigned int>(timeframes)}}, mDataSent{std::vector<bool>(timeframes), std::vector<bool>(timeframes)}, mTFStart{{firstTF, firstTF + timeframes}}, mTFEnd{{firstTF + timeframes - 1, mTFStart[1] + timeframes - 1}}, mSendPrecisetimeStamp{sendPrecisetimeStamp}
  {

    // pre calculate data description for output
    mDataDescrOut.reserve(mOutLanes);
    for (int i = 0; i < mOutLanes; ++i) {
      mDataDescrOut.emplace_back(getDataDescriptionIDC(i));
    }

    // sort vector for binary_search
    std::sort(mCRUs.begin(), mCRUs.end());

    for (auto& idcbuffer : mIDCs) {
      for (auto& idc : idcbuffer) {
        idc.resize(mTimeFrames);
      }
    }

    for (auto& processedCRUbuffer : mProcessedCRUs) {
      processedCRUbuffer.resize(mTimeFrames);
      for (auto& crusMap : processedCRUbuffer) {
        crusMap.reserve(mCRUs.size());
        for (const auto cruID : mCRUs) {
          crusMap.emplace(cruID, false);
        }
      }
    }
  };

  void init(o2::framework::InitContext& ic) final
  {
    mCheckMissingData = ic.options().get<bool>("check-for-missing-data");

    if (mLoadFromFile) {
      TFile fInp("IDCGroup.root", "READ");
      for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
        const auto key = dynamic_cast<TKey*>(keyAsObj);
        LOGP(info, "Key name: {} Type: {}", key->GetName(), key->GetClassName());

        if (std::strcmp(o2::tpc::IDCAverageGroup<o2::tpc::IDCAverageGroupCRU>::Class()->GetName(), key->GetClassName()) != 0) {
          LOGP(info, "skipping object. wrong class.");
          continue;
        }
        IDCAverageGroup<IDCAverageGroupCRU>* idcavg = (IDCAverageGroup<IDCAverageGroupCRU>*)fInp.Get(key->GetName());
        unsigned int cru = idcavg->getSector() * Mapper::NREGIONS + idcavg->getRegion();
        const std::vector<float>& idcData = idcavg->getIDCGroup().getData();
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
        delete idcavg;
      }
    }
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    LOGP(info, "finaliseCCDB");
    if (matcher == ConcreteDataMatcher("CTP", "ORBITRESET", 0)) {
      mOrbitResetTime = (*(std::vector<Long64_t>*)obj).front();
      LOGP(info, "Updating orbit reset from CCDB to {}", mOrbitResetTime);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // check which buffer to use for current incoming data
    const auto tf = processing_helpers::getCurrentTF(pc);

    // store precise timestamp for look up later
    if (mSendPrecisetimeStamp && pc.inputs().isValid("orbitreset")) {
      pc.inputs().get<std::vector<Long64_t>*>("orbitreset"); // check for new orbitReset
      if (pc.inputs().countValidInputs() == 1) {
        return;
      }
    }
    // automatically detect firstTF in case firstTF was not specified
    if (mTFStart.front() <= -1) {
      const auto firstTF = tf;
      const int offsetTF = std::abs(mTFStart.front() + 1);
      mTFStart = {firstTF + offsetTF, firstTF + offsetTF + mTimeFrames};
      mTFEnd = {mTFStart[1] - 1, mTFStart[1] - 1 + mTimeFrames};
      LOGP(info, "Setting {} as first TF", mTFStart[0]);
      LOGP(info, "Using offset of {} TFs for setting the first TF", offsetTF);
    }

    const bool currentBuffer = (tf > mTFEnd[mBuffer]) ? !mBuffer : mBuffer;
    if (mTFStart[currentBuffer] > tf) {
      LOGP(info, "all CRUs for current TF {} already received. Skipping this TF", tf);
      return;
    }

    const unsigned int currentOutLane = getOutLane(tf);
    const auto relTF = tf - mTFStart[currentBuffer];
    LOGP(info, "current TF: {}   relative TF: {}    current buffer: {}    current output lane: {}     mTFStart: {}", tf, relTF, currentBuffer, currentOutLane, mTFStart[currentBuffer]);

    if (relTF >= mProcessedCRU[currentBuffer].size()) {
      LOGP(warning, "Skipping tf {}: relative tf {} is larger than size of buffer: {}", tf, relTF, mProcessedCRU[currentBuffer].size());

      // check number of processed CRUs for previous TFs. If CRUs are missing for them, they are probably lost/not received
      if (mCheckMissingData) {
        checkMissingData(currentBuffer);
      }
      return;
    }

    if (!mLoadFromFile) {
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;

        // check if cru is specified in input cru list
        if (!(std::binary_search(mCRUs.begin(), mCRUs.end(), cru))) {
          LOGP(debug, "Received data from CRU: {} which was not specified as input. Skipping", cru);
          continue;
        }
        // to keep track of processed CRUs
        mProcessedCRUs[currentBuffer][relTF][cru] = true;

        // count total number of processed CRUs for given TF
        ++mProcessedCRU[currentBuffer][relTF];

        // storing IDCs
        mIDCs[currentBuffer][cru][relTF] = pc.inputs().get<pmr::vector<float>>(ref);
      }
    }

    LOGP(info, "number of received CRUs for current TF: {}    Needed a total number of processed CRUs of: {}   Current TF: {}", mProcessedCRU[currentBuffer][relTF], mCRUs.size(), tf);

    // check if all CRUs for current TF are already aggregated and send data
    if ((mProcessedCRU[currentBuffer][relTF] == mCRUs.size() || mLoadFromFile) && !mDataSent[currentBuffer][relTF]) {
      LOGP(info, "All data for current TF {} received. Sending data...", tf);
      mDataSent[currentBuffer][relTF] = true;
      ++mProcessedTFs[currentBuffer];
      sendOutput(pc, currentOutLane, currentBuffer, relTF);
    }

    if (mProcessedTFs[currentBuffer] == mTimeFrames) {
      LOGP(info, "All TFs {} for current buffer received. Clearing buffer", tf);
      clearBuffer(currentBuffer);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  /// return data description for aggregated IDCs for given lane
  static header::DataDescription getDataDescriptionIDC(const int lane)
  {
    const std::string name = fmt::format("IDCAGG{}", lane).data();
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

  static constexpr header::DataDescription getDataDescriptionIDCRelTF() { return header::DataDescription{"IDCRELTF"}; }
  static constexpr header::DataDescription getDataDescriptionIDCOrbitReset() { return header::DataDescription{"IDCORBITRESET"}; }

 private:
  std::vector<uint32_t> mCRUs{};                                                                                                                                                                         ///< CRUs to process in this instance
  const unsigned int mTimeFrames{};                                                                                                                                                                      ///< number of TFs per aggregation interval
  const unsigned int mOutLanes{};                                                                                                                                                                        ///< number of output lanes
  const bool mLoadFromFile{};                                                                                                                                                                            ///< if true data will be loaded from root file
  std::array<int, 2> mProcessedTFs{{0, 0}};                                                                                                                                                              ///< number of processed time frames to keep track of when the writing to CCDB will be done
  std::array<std::array<std::vector<pmr::vector<float>>, CRU::MaxCRU>, 2> mIDCs{};                                                                                                                       ///< grouped and integrated IDCs for the whole TPC. CRU -> time frame -> IDCs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::vector<unsigned int>, 2> mProcessedCRU{};                                                                                                                                              ///< counter of received data from CRUs per TF to merge incoming data from FLPs. Buffer used in case one FLP delivers the TF after the last TF for the current aggregation interval faster then the other FLPs the last TF.
  std::array<std::vector<bool>, 2> mDataSent{};                                                                                                                                                          ///< to keep track if the data for a given tf has already been sent
  std::array<std::vector<std::unordered_map<unsigned int, bool>>, 2> mProcessedCRUs{};                                                                                                                   ///< to keep track of the already processed CRUs ([buffer][relTF][CRU])
  std::array<long, 2> mTFStart{};                                                                                                                                                                        ///< storing of first TF for buffer interval
  std::array<long, 2> mTFEnd{};                                                                                                                                                                          ///< storing of last TF for buffer interval
  const bool mSendPrecisetimeStamp{true};                                                                                                                                                                ///< use precise time stamp when writing to CCDB
  unsigned int mCurrentOutLane{0};                                                                                                                                                                       ///< index for keeping track of the current output lane
  bool mBuffer{false};                                                                                                                                                                                   ///< buffer index
  bool mCheckMissingData{false};                                                                                                                                                                         ///< perform check for missing data
  Long64_t mOrbitResetTime{};                                                                                                                                                                            ///< orbit reset time to calculate precise time stamp
  const std::vector<InputSpec> mFilter = {{"idcsgroup", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescriptionIDCGroup()}, Lifetime::Timeframe}}; ///< filter for looping over input data
  std::vector<header::DataDescription> mDataDescrOut{};

  void sendOutput(o2::framework::ProcessingContext& pc, const unsigned int currentOutLane, const bool currentBuffer, const unsigned int relTF)
  {
    if (mSendPrecisetimeStamp) {
      pc.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionIDCOrbitReset(), header::DataHeader::SubSpecificationType{currentOutLane}}, mOrbitResetTime);
    }

    if (!mLoadFromFile) {
      for (unsigned int i = 0; i < mCRUs.size(); ++i) {
        pc.outputs().adoptContainer(Output{gDataOriginTPC, mDataDescrOut[currentOutLane], header::DataHeader::SubSpecificationType{mCRUs[i]}}, std::move(mIDCs[currentBuffer][mCRUs[i]][relTF]));
      }
    } else {
      for (unsigned int i = 0; i < mCRUs.size(); ++i) {
        pc.outputs().snapshot(Output{gDataOriginTPC, mDataDescrOut[currentOutLane], header::DataHeader::SubSpecificationType{mCRUs[i]}}, mIDCs[currentBuffer][mCRUs[i]][relTF]);
      }
    }
  }

  /// returns the output lane to which the data will be send
  unsigned int getOutLane(const uint32_t tf) const { return (tf > mTFEnd[mBuffer]) ? (mCurrentOutLane + 1) % mOutLanes : mCurrentOutLane; }

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
    std::fill(mDataSent[currentBuffer].begin(), mDataSent[currentBuffer].end(), false);

    // set integration range for next integration interval
    mTFStart[mBuffer] = mTFEnd[!mBuffer] + 1;
    mTFEnd[mBuffer] = mTFStart[mBuffer] + mTimeFrames - 1;

    // switch buffer
    mBuffer = !mBuffer;

    // set output lane
    mCurrentOutLane = ++mCurrentOutLane % mOutLanes;
  }

  void checkMissingData(const bool currentBuffer)
  {
    for (int iTF = 0; iTF < mProcessedCRU[currentBuffer].size(); ++iTF) {
      LOGP(info, "Checking rel TF: {} for missing CRUs", iTF);
      if (mProcessedCRU[currentBuffer][iTF] != mCRUs.size()) {
        LOGP(warning, "CRUs for TF: {} are missing!", iTF);

        // find actuall CRUs
        for (auto& it : mProcessedCRUs[currentBuffer][iTF]) {
          if (!it.second) {
            LOGP(warning, "Couldnt find data for CRU {} possibly not received! Setting dummy values...", it.first);
          }
        }
      }
    }
  }
};

DataProcessorSpec getTPCDistributeIDCSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int outlanes, const int firstTF, const bool loadFromFile, const bool sendPrecisetimeStamp = false)
{
  std::vector<InputSpec> inputSpecs;
  if (!loadFromFile) {
    inputSpecs.reserve(crus.size());
    for (const auto& cru : crus) {
      const header::DataHeader::SubSpecificationType subSpec{cru << 7};
      inputSpecs.emplace_back(InputSpec{"idcsgroup", gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescriptionIDCGroup(), subSpec, Lifetime::Timeframe});
    }
  }

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.reserve(outlanes);
  for (unsigned int lane = 0; lane < outlanes; ++lane) {
    const header::DataHeader::SubSpecificationType subSpec{lane};
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDC(lane)}, Lifetime::Sporadic);
  }

  if (sendPrecisetimeStamp && (ilane == 0)) {
    inputSpecs.emplace_back("orbitreset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset"));
    for (unsigned int lane = 0; lane < outlanes; ++lane) {
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCDistributeIDCSpec::getDataDescriptionIDCOrbitReset(), header::DataHeader::SubSpecificationType{lane}}, Lifetime::Sporadic);
    }
  }

  const auto id = fmt::format("tpc-distribute-idc-{:02}", ilane);
  DataProcessorSpec spec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCDistributeIDCSpec>(crus, timeframes, outlanes, loadFromFile, firstTF, (ilane == 0) ? sendPrecisetimeStamp : false)},
    Options{{"check-for-missing-data", VariantType::Bool, false, {"Perform check if all data is received."}}}}; // end DataProcessorSpec
  spec.rank = ilane;
  return spec;
}

} // namespace o2::tpc

#endif
