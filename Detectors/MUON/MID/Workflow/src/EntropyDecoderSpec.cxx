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
  std::array<std::vector<o2::mid::ROFRecord>, NEvTypes> rofs{};
  std::array<std::vector<o2::mid::ColumnData>, NEvTypes> cols{};

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::mid::CTF::getImage(buff.data());
    mCTFCoder.decode(ctfImage, rofs, cols);
  }

  for (uint32_t it = 0; it < NEvTypes; it++) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginMID, "DATA", it, Lifetime::Timeframe}, cols[it]);
    pc.outputs().snapshot(Output{o2::header::gDataOriginMID, "DATAROF", it, Lifetime::Timeframe}, rofs[it]);
  }

  mTimer.Stop();
  LOG(info) << "Decoded {" << cols[0].size() << ',' << cols[1].size() << ',' << cols[2].size()
            << "} MID columns for {" << rofs[0].size() << ',' << rofs[1].size() << ',' << rofs[2].size()
            << "} ROFRecords in " << mTimer.CpuTime() - cput << " s";
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

  return DataProcessorSpec{
    "mid-entropy-decoder",
    Inputs{InputSpec{"ctf", "MID", "CTFDATA", sspec, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "", {"CTF dictionary: empty=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace mid
} // namespace o2
