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
#include "CPVWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace cpv
{

EntropyDecoderSpec::EntropyDecoderSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("cpv-ctf-dictionary");
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Decoder);
  }
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& triggers = pc.outputs().make<std::vector<TriggerRecord>>(OutputRef{"triggers"});
  auto& clusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"clusters"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  const auto ctfImage = o2::cpv::CTF::getImage(buff.data());
  mCTFCoder.decode(ctfImage, triggers, clusters);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << clusters.size() << " CPV clusters in " << triggers.size() << " triggers in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "CPV Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec()
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"triggers"}, "CPV", "CLUSTERTRIGRECS", 0, Lifetime::Timeframe},
    OutputSpec{{"clusters"}, "CPV", "CLUSTERS", 0, Lifetime::Timeframe}};

  return DataProcessorSpec{
    "cpv-entropy-decoder",
    Inputs{InputSpec{"ctf", "CPV", "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{{"cpv-ctf-dictionary", VariantType::String, "ctf_dictionary.root", {"File of CTF decoding dictionary"}}}};
}

} // namespace cpv
} // namespace o2
