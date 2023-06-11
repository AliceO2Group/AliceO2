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
#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

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
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_TPC");

  auto& compclusters = pc.outputs().make<std::vector<char>>(OutputRef{"output"});
  if (buff.size()) {
    const auto ctfImage = o2::tpc::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, compclusters);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded " << buff.size() * sizeof(o2::ctf::BufferType) << " encoded bytes to "
            << compclusters.size() << " bytes, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TPC Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_TPC", "TPC", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict_TPC", "TPC", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TPC/Calib/CTFDictionaryTree"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    "tpc-entropy-decoder",
    inputs,
    Outputs{OutputSpec{{"output"}, "TPC", "COMPCLUSTERSFLAT", 0, Lifetime::Timeframe},
            OutputSpec{{"ctfrep"}, "TPC", "CTFDECREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace tpc
} // namespace o2
