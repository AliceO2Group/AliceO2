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

/// @file TPCFLPIDCSpec.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief TPC device for processing CMVs on FLPs

#ifndef O2_TPCFLPIDCSPEC_H
#define O2_TPCFLPIDCSPEC_H

#include <vector>
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
    LOGP(info, "Processing CMVs for TF {} for CRUs {} to {}", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back());

    ++mCountTFsForBuffer;

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification >> 7;
      auto vecCMVs = pc.inputs().get<o2::pmr::vector<float>>(ref);
      mCMVs[cru].insert(mCMVs[cru].end(), vecCMVs.begin(), vecCMVs.end());
    }

    if (mCountTFsForBuffer >= mNTFsBuffer) {
      mCountTFsForBuffer = 0;
      for (const auto cru : mCRUs) {
        LOGP(info, "Sending CMVs of size {} for TF {}", mCMVs[cru].size(), processing_helpers::getCurrentTF(pc));
        sendOutput(pc.outputs(), cru);
      }
    }

    if (mDumpCMVs) {
      TFile fOut(fmt::format("CMVs_{}_tf_{}.root", mLane, processing_helpers::getCurrentTF(pc)).data(), "RECREATE");
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        auto vec = pc.inputs().get<std::vector<float>>(ref);
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

  static constexpr header::DataDescription getDataDescriptionCMVGroup(const Side side) { return (side == Side::A) ? getDataDescriptionCMVGroupA() : getDataDescriptionCMVGroupC(); }
  static constexpr header::DataDescription getDataDescriptionCMVGroupA() { return header::DataDescription{"CMVGROUPA"}; }
  static constexpr header::DataDescription getDataDescriptionCMVGroupC() { return header::DataDescription{"CMVGROUPC"}; }

 private:
  const int mLane{};                                                ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                              ///< CRUs to process in this instance
  int mNTFsBuffer{1};                                               ///< number of TFs to buffer before sending
  bool mDumpCMVs{};                                                 ///< dump CMVs to file for debugging
  int mCountTFsForBuffer{0};                                        ///< counts TFs to track when to send output
  std::unordered_map<unsigned int, o2::pmr::vector<float>> mCMVs{}; ///< buffered CMV vectors per CRU
  const std::vector<InputSpec> mFilter = {{"cmvs", ConcreteDataTypeMatcher{gDataOriginTPC, "CMVVECTOR"}, Lifetime::Timeframe}};

  void sendOutput(DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    output.adoptContainer(Output{gDataOriginTPC, getDataDescriptionCMVGroup(CRU(cru).side()), subSpec}, std::move(mCMVs[cru]));
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
    inputSpecs.emplace_back(InputSpec{"cmvs", gDataOriginTPC, "CMVVECTOR", subSpec, Lifetime::Timeframe});
    const Side side = CRU(cru).side();
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup(side), subSpec}, Lifetime::Sporadic);
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