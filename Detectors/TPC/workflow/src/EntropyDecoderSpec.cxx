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
#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("tpc-ctf-dictionary");
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Decoder);
  }
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& compclusters = pc.outputs().make<std::vector<char>>(OutputRef{"output"});
  const auto ctfImage = o2::tpc::CTF::getImage(buff.data());
  mCTFCoder.decode(ctfImage, compclusters);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << buff.size() * sizeof(o2::ctf::BufferType) << " encoded bytes to "
            << compclusters.size() << " bytes in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TPC Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec()
{
  return DataProcessorSpec{
    "tpc-entropy-decoder",
    Inputs{InputSpec{"ctf", "TPC", "CTFDATA", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"output"}, "TPC", "COMPCLUSTERSFLAT", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{{"tpc-ctf-dictionary", VariantType::String, "ctf_dictionary.root", {"File of CTF decoding dictionary"}}}};
}

} // namespace tpc
} // namespace o2
