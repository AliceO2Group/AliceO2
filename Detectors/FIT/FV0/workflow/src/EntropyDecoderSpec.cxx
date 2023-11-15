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

/// @file   EntropyDecoderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FV0Workflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

EntropyDecoderSpec::EntropyDecoderSpec(int verbosity) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setDictBinding("ctfdict_FV0");
}

void EntropyDecoderSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  o2::ctf::CTFIOSize iosize;

  mCTFCoder.updateTimeDependentParams(pc, true);
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_FV0");

  auto& digits = pc.outputs().make<std::vector<o2::fv0::Digit>>(OutputRef{"digits"});
  auto& channels = pc.outputs().make<std::vector<o2::fv0::ChannelData>>(OutputRef{"channels"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::fv0::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, digits, channels);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded " << channels.size() << " FV0 channels in " << digits.size() << " digits, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "FV0 Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"digits"}, "FV0", "DIGITSBC", 0, Lifetime::Timeframe},
    OutputSpec{{"channels"}, "FV0", "DIGITSCH", 0, Lifetime::Timeframe},
    OutputSpec{{"ctfrep"}, "FV0", "CTFDECREP", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_FV0", "FV0", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict_FV0", "FV0", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("FV0/Calib/CTFDictionaryTree"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    "fv0-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

} // namespace fv0
} // namespace o2
