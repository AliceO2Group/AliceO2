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

/// @file TPCFLPCMVSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief TPC device for processing CMVs on FLPs

#ifndef O2_TPCFLPIDCSPEC_H
#define O2_TPCFLPIDCSPEC_H

#include <vector>
#include <unordered_map>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/CRU.h"
#include "TFile.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFLPCMVDevice : public o2::framework::Task
{
 public:
  TPCFLPCMVDevice(const int lane, const std::vector<uint32_t>& crus, const int nTFsBuffer)
    : mLane{lane}, mCRUs{crus}, mNTFsBuffer{nTFsBuffer} {}

  void init(o2::framework::InitContext& ic) final
  {
    mDumpCMVs = ic.options().get<bool>("dump-cmvs-flp");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(debug, "Processing CMVs for TF {} for CRUs {} to {}", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back());

    ++mCountTFsForBuffer;

    // Capture heartbeatOrbit / heartbeatBC from the first TF in the buffer
    if (mCountTFsForBuffer == 1) {
      for (auto& ref : InputRecordWalker(pc.inputs(), mOrbitFilter)) {
        auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const uint32_t cru = hdr->subSpecification >> 7;
        if (mFirstOrbitBC.find(cru) == mFirstOrbitBC.end()) {
          auto orbitVec = pc.inputs().get<std::vector<uint64_t>>(ref);
          if (!orbitVec.empty()) {
            mFirstOrbitBC[cru] = orbitVec[0]; // packed: orbit<<32 | bc
          }
        }
      }
    }

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification >> 7;
      auto vecCMVs = pc.inputs().get<o2::pmr::vector<uint16_t>>(ref);
      mCMVs[cru].insert(mCMVs[cru].end(), vecCMVs.begin(), vecCMVs.end());
    }

    if (mCountTFsForBuffer >= mNTFsBuffer) {
      mCountTFsForBuffer = 0;
      for (const auto cru : mCRUs) {
        LOGP(debug, "Sending CMVs of size {} for TF {}", mCMVs[cru].size(), processing_helpers::getCurrentTF(pc));
        sendOutput(pc.outputs(), cru);
      }
      mFirstOrbitBC.clear();
    }

    if (mDumpCMVs) {
      TFile fOut(fmt::format("CMVs_{}_tf_{}.root", mLane, processing_helpers::getCurrentTF(pc)).data(), "RECREATE");
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        auto vec = pc.inputs().get<std::vector<uint16_t>>(ref);
        fOut.WriteObject(&vec, fmt::format("CRU_{}", cru).data());
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (mCountTFsForBuffer > 0) {
      LOGP(info, "Flushing remaining {} buffered TFs at end of stream", mCountTFsForBuffer);
      for (const auto cru : mCRUs) {
        sendOutput(ec.outputs(), cru);
      }
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionCMVGroup() { return header::DataDescription{"CMVGROUP"}; }

  /// Data description for the packed (orbit<<32|bc) scalar forwarded alongside each CRU's CMVGROUP.
  static constexpr header::DataDescription getDataDescriptionCMVOrbitInfo() { return header::DataDescription{"CMVORBITINFO"}; }

 private:
  const int mLane{};                                                   ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                                 ///< CRUs to process in this instance
  int mNTFsBuffer{1};                                                  ///< number of TFs to buffer before sending
  bool mDumpCMVs{};                                                    ///< dump CMVs to file for debugging
  int mCountTFsForBuffer{0};                                           ///< counts TFs to track when to send output
  std::unordered_map<unsigned int, o2::pmr::vector<uint16_t>> mCMVs{}; ///< buffered raw 16-bit CMV values per CRU
  std::unordered_map<uint32_t, uint64_t> mFirstOrbitBC{};              ///< first packed orbit/BC per CRU for the current buffer window

  /// Filter for CMV float vectors (one CMVVECTOR message per CRU per TF)
  const std::vector<InputSpec> mFilter = {{"cmvs", ConcreteDataTypeMatcher{gDataOriginTPC, "CMVVECTOR"}, Lifetime::Timeframe}};
  /// Filter for CMV packet timing info (one CMVORBITS message per CRU per TF, sent by CMVToVectorSpec)
  const std::vector<InputSpec> mOrbitFilter = {{"cmvorbits", ConcreteDataTypeMatcher{gDataOriginTPC, "CMVORBITS"}, Lifetime::Timeframe}};

  void sendOutput(DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};

    // Forward the first-TF orbit/BC for this CRU (0 if unavailable for any reason)
    uint64_t orbitBC = 0;
    if (auto it = mFirstOrbitBC.find(cru); it != mFirstOrbitBC.end()) {
      orbitBC = it->second;
    }
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionCMVOrbitInfo(), subSpec}, orbitBC);

    output.adoptContainer(Output{gDataOriginTPC, getDataDescriptionCMVGroup(), subSpec}, std::move(mCMVs[cru]));
  }
};

DataProcessorSpec getTPCFLPCMVSpec(const int ilane, const std::vector<uint32_t>& crus, const int nTFsBuffer = 1)
{
  std::vector<OutputSpec> outputSpecs;
  std::vector<InputSpec> inputSpecs;
  outputSpecs.reserve(crus.size());
  inputSpecs.reserve(crus.size());

  for (const auto& cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};

    // Inputs from CMVToVectorSpec
    inputSpecs.emplace_back(InputSpec{"cmvs", gDataOriginTPC, "CMVVECTOR", subSpec, Lifetime::Timeframe});
    inputSpecs.emplace_back(InputSpec{"cmvorbits", gDataOriginTPC, "CMVORBITS", subSpec, Lifetime::Timeframe});

    // Outputs to TPCDistributeCMVSpec
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup(), subSpec}, Lifetime::Sporadic);
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo(), subSpec}, Lifetime::Sporadic);
  }

  const auto id = fmt::format("tpc-flp-cmv-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFLPCMVDevice>(ilane, crus, nTFsBuffer)},
    Options{{"dump-cmvs-flp", VariantType::Bool, false, {"Dump CMVs to file"}}}};
}

} // namespace o2::tpc
#endif