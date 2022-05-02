// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "Framework/CCDBParamSpec.h"
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
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::mch::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

EntropyEncoderSpec::EntropyEncoderSpec() : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder)
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyEncoderSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

void EntropyEncoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  mCTFCoder.updateTimeDependentParams(pc);
  auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
  auto digits = pc.inputs().get<gsl::span<o2::mch::Digit>>("digits");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"MCH", "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, rofs, digits);
  auto sz = mCTFCoder.finaliseCTFOutput<CTF>(buffer);
  mTimer.Stop();
  LOG(info) << fmt::format("Created encoded data ({} digits and {} rofs) of size {} ({:5.1f} MB) for MCH in {:5.1f} s ",
                           digits.size(), rofs.size(), sz, sz / 1024.0 / 1024, mTimer.CpuTime() - cput);
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "MCH Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(const char* specName)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("rofs", "MCH", "DIGITROFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digits", "MCH", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "MCH", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/CTFDictionary"));

  return DataProcessorSpec{
    specName,
    inputs,
    Outputs{{"MCH", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
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
  wf.emplace_back(o2::mch::getEntropyEncoderSpec("mch-entropy-encoder"));
  return wf;
}
