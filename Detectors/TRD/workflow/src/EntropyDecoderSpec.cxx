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
#include "TRDWorkflow/EntropyDecoderSpec.h"
#include "TRDReconstruction/CTFCoder.h"
#include <TStopwatch.h>

using namespace o2::framework;

namespace o2
{
namespace trd
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
  o2::trd::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

EntropyDecoderSpec::EntropyDecoderSpec(int verbosity) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
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

  mCTFCoder.updateTimeDependentParams(pc);
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& triggers = pc.outputs().make<std::vector<TriggerRecord>>(OutputRef{"triggers"});
  auto& tracklets = pc.outputs().make<std::vector<Tracklet64>>(OutputRef{"tracklets"});
  auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::trd::CTF::getImage(buff.data());
    mCTFCoder.decode(ctfImage, triggers, tracklets, digits);
  }

  mTimer.Stop();
  LOG(info) << "Decoded " << tracklets.size() << " TRD tracklets and " << digits.size() << " digits in " << triggers.size() << " triggers in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"triggers"}, "TRD", "TRKTRGRD", 0, Lifetime::Timeframe},
    OutputSpec{{"tracklets"}, "TRD", "TRACKLETS", 0, Lifetime::Timeframe},
    OutputSpec{{"digits"}, "TRD", "DIGITS", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf", "TRD", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "TRD", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CTFDictionary"));

  return DataProcessorSpec{
    "trd-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace trd
} // namespace o2
