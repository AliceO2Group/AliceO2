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
#include "EMCALWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace emcal
{

EntropyDecoderSpec::EntropyDecoderSpec(int verbosity, unsigned int sspecOut) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder), mSSpecOut(sspecOut)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setSupportBCShifts(true);
  mCTFCoder.setDictBinding("ctfdict_EMC");
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
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_EMC");

  auto& triggers = pc.outputs().make<std::vector<TriggerRecord>>(OutputRef{"triggers", mSSpecOut});
  auto& cells = pc.outputs().make<std::vector<Cell>>(OutputRef{"cells", mSSpecOut});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::emcal::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, triggers, cells);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded " << cells.size() << " EMCAL cells in " << triggers.size() << " triggers, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "EMCAL Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspecInp, unsigned int sspecOut)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"triggers"}, "EMC", "CELLSTRGR", sspecOut, Lifetime::Timeframe},
    OutputSpec{{"cells"}, "EMC", "CELLS", sspecOut, Lifetime::Timeframe},
    OutputSpec{{"ctfrep"}, "EMC", "CTFDECREP", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_EMC", "EMC", "CTFDATA", sspecInp, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict_EMC", "EMC", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("EMC/Calib/CTFDictionaryTree"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    "emcal-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity, sspecOut)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

} // namespace emcal
} // namespace o2
