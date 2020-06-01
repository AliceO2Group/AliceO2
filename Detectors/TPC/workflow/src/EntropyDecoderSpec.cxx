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
#include "TPCReconstruction/CTFCoder.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void EntropyDecoderSpec::init(InitContext& ic)
{
  // at the moment do nothing
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& compclusters = pc.outputs().make<std::vector<char>>(OutputRef{"output"});
  const auto ctfImage = o2::tpc::CTF::getImage(buff.data());
  CTFCoder::decode(ctfImage, compclusters);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << buff.size() * sizeof(o2::ctf::BufferType) << " encoded bytes to "
            << compclusters.size() << " bytes in" << mTimer.CpuTime() - cput << "\n";
}

DataProcessorSpec getEntropyDecoderSpec()
{
  return DataProcessorSpec{
    "TPC",
    Inputs{InputSpec{"ctf", "TPC", "CTFDATA", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"output"}, "TPC", "COMPCLUSTERSFLAT", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{}};
}

} // namespace tpc
} // namespace o2
