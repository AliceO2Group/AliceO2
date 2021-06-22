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
#include "TRDWorkflow/EntropyDecoderSpec.h"
#include "TRDReconstruction/CTFCoder.h"
#include <TStopwatch.h>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class EntropyDecoderSpec : public o2::framework::Task
{
 public:
  EntropyDecoderSpec();
  ~EntropyDecoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  o2::trd::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

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

  auto& triggers = pc.outputs().make<std::vector<TriggerRecord>>(OutputRef{"triggers"});
  auto& tracklets = pc.outputs().make<std::vector<Tracklet64>>(OutputRef{"tracklets"});
  auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  const auto ctfImage = o2::trd::CTF::getImage(buff.data());
  mCTFCoder.decode(ctfImage, triggers, tracklets, digits);

  mTimer.Stop();
  LOG(INFO) << "Decoded " << tracklets.size() << " TRD tracklets and " << digits.size() << " digits in " << triggers.size() << " triggers in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TRD Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyDecoderSpec()
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"triggers"}, "TRD", "TRKTRGRD", 0, Lifetime::Timeframe},
    OutputSpec{{"tracklets"}, "TRD", "TRACKLETS", 0, Lifetime::Timeframe},
    OutputSpec{{"digits"}, "TRD", "DIGITS", 0, Lifetime::Timeframe}};

  return DataProcessorSpec{
    "trd-entropy-decoder",
    Inputs{InputSpec{"ctf", "TRD", "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF decoding dictionary"}}}};
}

} // namespace trd
} // namespace o2
