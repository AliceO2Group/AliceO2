// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "MCHCTF/CTFCoder.h"
#include <TStopwatch.h>
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace mch
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec();
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  o2::mch::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

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
  auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
  auto digits = pc.inputs().get<gsl::span<o2::mch::Digit>>("digits");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"MCH", "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, rofs, digits);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  //  eeb->print();
  mTimer.Stop();
  LOG(INFO) << fmt::format("Created encoded data ({} digits and {} rofs) of size {} ({:5.1f} MB) for MCH in {:5.1f} s ",
                           digits.size(), rofs.size(), eeb->size(), eeb->size() / 1024.0 / 1024, mTimer.CpuTime() - cput);
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "MCH Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("rofs", "MCH", "DIGITROFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digits", "MCH", "DIGITS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "mch-entropy-encoder",
    inputs,
    Outputs{{"MCH", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"Path to pre-computed CTF encoding dictionary to be used for encoding"}}}};
}

} // namespace mch
} // namespace o2

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(o2::mch::getEntropyEncoderSpec());
  return wf;
}
