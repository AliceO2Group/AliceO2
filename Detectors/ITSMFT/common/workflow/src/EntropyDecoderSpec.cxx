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
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h" // RS DUMMY, being outphased, to remove
#include "ITSMFTReconstruction/CTFCoder.h"
#include "ITSMFTWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

EntropyDecoderSpec::EntropyDecoderSpec(o2::header::DataOrigin orig) : mOrigin(orig)
{
  assert(orig == o2::header::gDataOriginITS || orig == o2::header::gDataOriginMFT);
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");

  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(OutputRef{"ROframes"});
  auto& compcl = pc.outputs().make<std::vector<o2::itsmft::CompClusterExt>>(OutputRef{"compClusters"});
  auto& patterns = pc.outputs().make<std::vector<unsigned char>>(OutputRef{"patterns"});
  auto& fullDummy = pc.outputs().make<std::vector<o2::itsmft::Cluster>>(OutputRef{"fullclusters"}); // RS DUMMY, being outphased

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  const auto ctfImage = o2::itsmft::CTF::getImage(buff.data());
  CTFCoder::decode(ctfImage, rofs, compcl, patterns);

  LOG(INFO) << "Decoded " << compcl.size() << " clusters in " << rofs.size() << " RO frames";
}

DataProcessorSpec getEntropyDecoderSpec(o2::header::DataOrigin orig)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"compClusters"}, orig, "COMPCLUSTERS", 0, Lifetime::Timeframe},
    OutputSpec{{"patterns"}, orig, "PATTERNS", 0, Lifetime::Timeframe},
    OutputSpec{{"ROframes"}, orig, orig == o2::header::gDataOriginITS ? "ITSClusterROF" : "MFTClusterROF", 0, Lifetime::Timeframe},
    OutputSpec{{"fullclusters"}, orig, "CLUSTERS", 0, Lifetime::Timeframe} // RS DUMMY, being outphased
  };

  return DataProcessorSpec{
    orig == o2::header::gDataOriginITS ? "its-entropy-decoder" : "mft-entropy-decoder",
    Inputs{InputSpec{"ctf", orig, "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(orig)},
    Options{}};
}

} // namespace itsmft
} // namespace o2
