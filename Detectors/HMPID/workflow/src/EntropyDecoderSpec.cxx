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
#include "HMPIDWorkflow/EntropyDecoderSpec.h"
#include "HMPIDReconstruction/CTFCoder.h"
#include <TStopwatch.h>

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

class EntropyDecoderSpec : public o2::framework::Task
{
 public:
  EntropyDecoderSpec(int verbosity);
  ~EntropyDecoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::hmpid::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

EntropyDecoderSpec::EntropyDecoderSpec(int verbosity) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setSupportBCShifts(true);
  mCTFCoder.setDictBinding("ctfdict_HMP");
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

  mCTFCoder.updateTimeDependentParams(pc);
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_HMP");

  auto& triggers = pc.outputs().make<std::vector<Trigger>>(OutputRef{"triggers"});
  auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::hmpid::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, triggers, digits);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded " << digits.size() << " HMPID digits in " << triggers.size() << " triggers, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "HMPID Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"triggers"}, "HMP", "INTRECORDS", 0, Lifetime::Timeframe},
    OutputSpec{{"digits"}, "HMP", "DIGITS", 0, Lifetime::Timeframe},
    OutputSpec{{"ctfrep"}, "HMP", "CTFDECREP", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_HMP", "HMP", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict_HMP", "HMP", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("HMP/Calib/CTFDictionary"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    "hmpid-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace hmpid
} // namespace o2
