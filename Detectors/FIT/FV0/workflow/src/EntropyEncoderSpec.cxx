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
#include "FV0Workflow/EntropyEncoderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
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
  auto digits = pc.inputs().get<gsl::span<o2::fv0::BCData>>("digits");
  auto channels = pc.inputs().get<gsl::span<o2::fv0::ChannelData>>("channels");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"FV0", "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, digits, channels);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  mTimer.Stop();
  LOG(INFO) << "Created encoded data of size " << eeb->size() << " for FV0 in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "FV0 Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "FV0", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("channels", "FV0", "DIGITSCH", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "fv0-entropy-encoder",
    inputs,
    Outputs{{"FV0", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF encoding dictionary"}}}};
}

} // namespace fv0
} // namespace o2
