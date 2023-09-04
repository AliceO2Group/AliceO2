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

#include "MIDWorkflow/EntropyDecoderSpec.h"

#include <vector>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace mid
{

EntropyDecoderSpec::EntropyDecoderSpec(int verbosity) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setDictBinding("ctfdict_MID");
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
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_MID");

  std::array<std::vector<o2::mid::ROFRecord>, NEvTypes> rofs{};
  std::array<std::vector<o2::mid::ColumnData>, NEvTypes> cols{};
  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::mid::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, rofs, cols);
  }
  size_t insize = 0;
  for (uint32_t it = 0; it < NEvTypes; it++) {
    insize += cols[it].size() * sizeof(o2::mid::ColumnData);
    pc.outputs().snapshot(Output{o2::header::gDataOriginMID, "DATA", it, Lifetime::Timeframe}, cols[it]);
    insize += rofs[it].size() * sizeof(o2::mid::ROFRecord);
    pc.outputs().snapshot(Output{o2::header::gDataOriginMID, "DATAROF", it, Lifetime::Timeframe}, rofs[it]);
  }
  iosize.rawIn = insize;
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded {" << cols[0].size() << ',' << cols[1].size() << ',' << cols[2].size()
            << "} MID columns for {" << rofs[0].size() << ',' << rofs[1].size() << ',' << rofs[2].size()
            << "} ROFRecords, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "MID Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs;
  for (o2::header::DataHeader::SubSpecificationType subSpec = 0; subSpec < NEvTypes; ++subSpec) {
    outputs.emplace_back(OutputSpec{header::gDataOriginMID, "DATA", subSpec});
    outputs.emplace_back(OutputSpec{header::gDataOriginMID, "DATAROF", subSpec});
  }
  outputs.emplace_back(OutputSpec{{"ctfrep"}, "MID", "CTFDECREP", 0, Lifetime::Timeframe});
  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_MID", "MID", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict_MID", "MID", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("MID/Calib/CTFDictionaryTree"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    "mid-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

} // namespace mid
} // namespace o2
