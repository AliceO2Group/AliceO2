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
#include "ZDCWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

EntropyDecoderSpec::EntropyDecoderSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("ctf-dict");
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Decoder);
  }
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& bcdata = pc.outputs().make<std::vector<o2::zdc::BCData>>(OutputRef{"trig"});
  auto& chans = pc.outputs().make<std::vector<o2::zdc::ChannelData>>(OutputRef{"chan"});
  auto& peds = pc.outputs().make<std::vector<o2::zdc::OrbitData>>(OutputRef{"peds"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  const auto ctfImage = o2::zdc::CTF::getImage(buff.data());
  mCTFCoder.decode(ctfImage, bcdata, chans, peds);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << chans.size() << " ZDC channels in " << bcdata.size() << " triggers and " << peds.size() << " pedestals in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "ZDC Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec()
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"trig"}, "ZDC", "DIGITSBC", 0, Lifetime::Timeframe},
    OutputSpec{{"chan"}, "ZDC", "DIGITSCH", 0, Lifetime::Timeframe},
    OutputSpec{{"peds"}, "ZDC", "DIGITSPD", 0, Lifetime::Timeframe}};

  return DataProcessorSpec{
    "zdc-entropy-decoder",
    Inputs{InputSpec{"ctf", "ZDC", "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF decoding dictionary"}}}};
}

} // namespace zdc
} // namespace o2
