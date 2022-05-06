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
#include "ZDCWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

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

  auto& bcdata = pc.outputs().make<std::vector<o2::zdc::BCData>>(OutputRef{"trig"});
  auto& chans = pc.outputs().make<std::vector<o2::zdc::ChannelData>>(OutputRef{"chan"});
  auto& peds = pc.outputs().make<std::vector<o2::zdc::OrbitData>>(OutputRef{"peds"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::zdc::CTF::getImage(buff.data());
    mCTFCoder.decode(ctfImage, bcdata, chans, peds);
  }
  mTimer.Stop();
  LOG(info) << "Decoded " << chans.size() << " ZDC channels in " << bcdata.size() << " triggers and " << peds.size() << " pedestals in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ZDC Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"trig"}, "ZDC", "DIGITSBC", 0, Lifetime::Timeframe},
    OutputSpec{{"chan"}, "ZDC", "DIGITSCH", 0, Lifetime::Timeframe},
    OutputSpec{{"peds"}, "ZDC", "DIGITSPD", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf", "ZDC", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "ZDC", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("ZDC/Calib/CTFDictionary"));

  return DataProcessorSpec{
    "zdc-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace zdc
} // namespace o2
