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

#ifndef O2_TPCFLPCMVSPEC_H
#define O2_TPCFLPCMVSPEC_H

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
#include "DataFormatsTPC/CMV.h"
#include "TFile.h"

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
    mEnableTrigger = ic.options().get<bool>("trigger");
    mTriggerThresholdCMV = ic.options().get<float>("trigger-threshold-cmv");
    mTriggerThresholdMeanMax = ic.options().get<float>("trigger-threshold-cmvMeanMax");
    mTriggerThresholdMeanMin = ic.options().get<float>("trigger-threshold-cmvMeanMin");
    mTriggerTimebinMin = ic.options().get<int>("trigger-threshold-timebinMin");
    mTriggerTimebinMax = ic.options().get<int>("trigger-threshold-timebinMax");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(debug, "Processing CMVs for TF {} for CRUs {} to {}", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back());

    ++mCountTFsForBuffer;

    // Capture heartbeatOrbit / heartbeatBC from the first TF in the buffer
    if (mCountTFsForBuffer == 1) {
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mOrbitFilter)) {
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

    bool triggered = false;
    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const uint32_t cru = tpcCRUHeader->subSpecification >> 7;
      auto vecCMVs = pc.inputs().get<o2::pmr::vector<uint16_t>>(ref);
      mCMVs[cru].insert(mCMVs[cru].end(), vecCMVs.begin(), vecCMVs.end());

      const bool cruTriggered = mEnableTrigger && evaluateTrigger(vecCMVs);
      triggered |= cruTriggered;
    }
    const header::DataHeader::SubSpecificationType trigSubSpec{mCRUs.front() << 7};
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVTrigger(), trigSubSpec}, triggered);

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
      for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
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
    ec.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionCMVGroup() { return header::DataDescription{"CMVGROUP"}; }

  /// Data description for the packed (orbit<<32|bc) scalar forwarded alongside each CRU's CMVGROUP.
  static constexpr header::DataDescription getDataDescriptionCMVOrbitInfo() { return header::DataDescription{"CMVORBITINFO"}; }

  /// Data description for the per-CRU per-TF trigger flag (empty span = not triggered or disabled; {1} = triggered).
  static constexpr header::DataDescription getDataDescriptionCMVTrigger() { return header::DataDescription{"CMVTRIGGER"}; }

 private:
  const int mLane{};                                                   ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                                 ///< CRUs to process in this instance
  int mNTFsBuffer{1};                                                  ///< number of TFs to buffer before sending
  bool mDumpCMVs{};                                                    ///< dump CMVs to file for debugging
  int mCountTFsForBuffer{0};                                           ///< counts TFs to track when to send output
  std::unordered_map<unsigned int, o2::pmr::vector<uint16_t>> mCMVs{}; ///< buffered raw 16-bit CMV values per CRU
  std::unordered_map<uint32_t, uint64_t> mFirstOrbitBC{};              ///< first packed orbit/BC per CRU for the current buffer window
  bool mEnableTrigger{false};                                          ///< enable CMV trigger evaluation
  float mTriggerThresholdCMV{-10.f};                                   ///< CMV value threshold: trigger sequence starts when value drops below this
  float mTriggerThresholdMeanMax{-40.f};                               ///< upper bound on trigger-sequence mean CMV value
  float mTriggerThresholdMeanMin{-80.f};                               ///< lower bound on trigger-sequence mean CMV value
  int mTriggerTimebinMin{4};                                           ///< minimum trigger-sequence length (timebins) to accept
  int mTriggerTimebinMax{-1};                                          ///< maximum trigger-sequence length (timebins) to accept; -1 disables

  /// Filter for CMV float vectors (one CMVVECTOR message per CRU per TF)
  const std::vector<o2::framework::InputSpec> mFilter = {{"cmvs", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CMVVECTOR"}, o2::framework::Lifetime::Timeframe}};
  /// Filter for CMV packet timing info (one CMVORBITS message per CRU per TF, sent by CMVToVectorSpec)
  const std::vector<o2::framework::InputSpec> mOrbitFilter = {{"cmvorbits", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CMVORBITS"}, o2::framework::Lifetime::Timeframe}};

  // Scan a CRU's CMV vector for contiguous below-threshold sequences.
  // Returns true as soon as one sequence satisfies both the length and mean criteria.
  bool evaluateTrigger(const o2::pmr::vector<uint16_t>& cmvs) const
  {
    float seqSum = 0.f;
    int seqLen = 0;

    auto checkSequence = [&]() -> bool {
      if (seqLen == 0) {
        return false;
      }
      const float mean = seqSum / seqLen;
      return (seqLen >= mTriggerTimebinMin) &&
             (mTriggerTimebinMax < 0 || seqLen <= mTriggerTimebinMax) &&
             (mean >= mTriggerThresholdMeanMin) &&
             (mean <= mTriggerThresholdMeanMax);
    };

    for (const auto raw : cmvs) {
      const float val = cmv::Data{raw}.getCMVFloat();
      if (val < mTriggerThresholdCMV) {
        seqSum += val;
        ++seqLen;
      } else {
        if (checkSequence()) {
          return true;
        }
        seqLen = 0;
        seqSum = 0.f;
      }
    }
    return checkSequence(); // trailing sequence that reached end of buffer
  }

  void sendOutput(o2::framework::DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};

    // Forward the first-TF orbit/BC for this CRU (0 if unavailable for any reason)
    uint64_t orbitBC = 0;
    if (auto it = mFirstOrbitBC.find(cru); it != mFirstOrbitBC.end()) {
      orbitBC = it->second;
    }
    output.snapshot(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVOrbitInfo(), subSpec}, orbitBC);

    output.adoptContainer(o2::framework::Output{o2::header::gDataOriginTPC, getDataDescriptionCMVGroup(), subSpec}, std::move(mCMVs[cru]));
  }
};

o2::framework::DataProcessorSpec getTPCFLPCMVSpec(const int ilane, const std::vector<uint32_t>& crus, const int nTFsBuffer = 1)
{
  std::vector<o2::framework::OutputSpec> outputSpecs;
  std::vector<o2::framework::InputSpec> inputSpecs;
  outputSpecs.reserve(crus.size() * 2 + 1);
  inputSpecs.reserve(crus.size() * 2);

  for (const auto& cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};

    // Inputs from CMVToVectorSpec
    inputSpecs.emplace_back(o2::framework::InputSpec{"cmvs", o2::header::gDataOriginTPC, "CMVVECTOR", subSpec, o2::framework::Lifetime::Timeframe});
    inputSpecs.emplace_back(o2::framework::InputSpec{"cmvorbits", o2::header::gDataOriginTPC, "CMVORBITS", subSpec, o2::framework::Lifetime::Timeframe});

    // Outputs to TPCDistributeCMVSpec
    outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVGroup(), subSpec}, o2::framework::Lifetime::Sporadic);
    outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVOrbitInfo(), subSpec}, o2::framework::Lifetime::Sporadic);
  }

  // Single per-FLP trigger output, subspec keyed on the first CRU
  const header::DataHeader::SubSpecificationType trigSubSpec{crus.front() << 7};
  outputSpecs.emplace_back(o2::framework::ConcreteDataMatcher{o2::header::gDataOriginTPC, TPCFLPCMVDevice::getDataDescriptionCMVTrigger(), trigSubSpec}, o2::framework::Lifetime::Timeframe);

  const auto id = fmt::format("tpc-flp-cmv-{:02}", ilane);
  return o2::framework::DataProcessorSpec{
    id.data(),
    inputSpecs,
    outputSpecs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<TPCFLPCMVDevice>(ilane, crus, nTFsBuffer)},
    o2::framework::Options{
      {"dump-cmvs-flp", o2::framework::VariantType::Bool, false, {"Dump CMVs to file"}},
      {"trigger", o2::framework::VariantType::Bool, false, {"Enable CMV trigger evaluation"}},
      {"trigger-threshold-cmv", o2::framework::VariantType::Float, -10.f, {"CMV threshold: sequence starts when value drops below this (ADC units)"}},
      {"trigger-threshold-cmvMeanMax", o2::framework::VariantType::Float, -40.f, {"Upper bound on trigger-sequence mean CMV value"}},
      {"trigger-threshold-cmvMeanMin", o2::framework::VariantType::Float, -80.f, {"Lower bound on trigger-sequence mean CMV value"}},
      {"trigger-threshold-timebinMin", o2::framework::VariantType::Int, 4, {"Minimum trigger-sequence length in timebins"}},
      {"trigger-threshold-timebinMax", o2::framework::VariantType::Int, -1, {"Maximum trigger-sequence length in timebins (-1 disables upper bound)"}}}};
}

} // namespace o2::tpc
#endif
