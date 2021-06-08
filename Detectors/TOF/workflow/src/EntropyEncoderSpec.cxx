// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EntropyEncoderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TOFBase/Digit.h"
#include "TOFWorkflowUtils/EntropyEncoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

EntropyEncoderSpec::EntropyEncoderSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyEncoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("ctf-dict");
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Encoder);
  }
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto compDigits = pc.inputs().get<gsl::span<Digit>>("compDigits");
  auto pspan = pc.inputs().get<gsl::span<uint8_t>>("patterns");
  auto rofs = pc.inputs().get<gsl::span<ReadoutWindowData>>("ROframes");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{o2::header::gDataOriginTOF, "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, rofs, compDigits, pspan);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  //  eeb->print();
  mTimer.Stop();
  LOG(INFO) << "Created encoded data of size " << eeb->size() << " for TOF in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TOF Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compDigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-entropy-encoder",
    inputs,
    Outputs{{o2::header::gDataOriginTOF, "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF encoding dictionary"}}}};
}

} // namespace tof
} // namespace o2
