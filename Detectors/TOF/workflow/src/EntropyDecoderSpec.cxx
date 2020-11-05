// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EntropyDecoderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TOFWorkflowUtils/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

EntropyDecoderSpec::EntropyDecoderSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("tof-ctf-dictionary");
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Decoder);
  }
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& digitheader = pc.outputs().make<DigitHeader>(OutputRef{"digitheader"});
  auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});
  auto& row = pc.outputs().make<std::vector<ReadoutWindowData>>(OutputRef{"row"});
  auto& patterns = pc.outputs().make<std::vector<uint32_t>>(OutputRef{"patterns"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  const auto ctfImage = o2::tof::CTF::getImage(buff.data());
  mCTFCoder.decode(ctfImage, row, digits, patterns);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << digits.size() << " digits in " << row.size() << " ROF in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TOF Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec()
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"digitheader"}, o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe},
    OutputSpec{{"digits"}, o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe},
    OutputSpec{{"row"}, o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe},
    OutputSpec{{"patterns"}, o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe}};

  return DataProcessorSpec{
    "tof-entropy-decoder",
    Inputs{InputSpec{"ctf", o2::header::gDataOriginTOF, "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{{"tof-ctf-dictionary", VariantType::String, "ctf_dictionary.root", {"File of CTF decoding dictionary"}}}};
}

} // namespace tof
} // namespace o2
