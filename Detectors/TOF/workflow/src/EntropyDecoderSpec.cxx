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
#include "TOFWorkflowUtils/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
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

  auto& digitheader = pc.outputs().make<DigitHeader>(OutputRef{"digitheader"});
  auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});
  auto& row = pc.outputs().make<std::vector<ReadoutWindowData>>(OutputRef{"row"});
  auto& patterns = pc.outputs().make<std::vector<uint8_t>>(OutputRef{"patterns"});
  //  auto& diagnostic = pc.outputs().make<o2::tof::Diagnostic>(OutputRef{"diafreq"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::tof::CTF::getImage(buff.data());
    mCTFCoder.decode(ctfImage, row, digits, patterns);
  }

  // fill diagnostic frequencies
  mFiller.clearCounts();
  for (auto digit : digits) {
    mFiller.addCount(digit.getChannel());
  }
  mFiller.setReadoutWindowData(row, patterns);
  mFiller.fillDiagnosticFrequency();
  auto diagnostic = mFiller.getDiagnosticFrequency();
  auto creationTime = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
  diagnostic.setTimeStamp(creationTime / 1000);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIAFREQ", 0, Lifetime::Timeframe}, diagnostic);

  mTimer.Stop();
  LOG(info) << "Decoded " << digits.size() << " digits in " << row.size() << " ROF in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TOF Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"digitheader"}, o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe},
    OutputSpec{{"digits"}, o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe},
    OutputSpec{{"row"}, o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe},
    OutputSpec{{"patterns"}, o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe},
    OutputSpec{{"diafreq"}, o2::header::gDataOriginTOF, "DIAFREQ", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf", "TOF", "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "TOF", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/CTFDictionary"));

  return DataProcessorSpec{
    "tof-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}}}};
}

} // namespace tof
} // namespace o2
