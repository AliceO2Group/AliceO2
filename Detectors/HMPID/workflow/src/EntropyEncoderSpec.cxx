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

/// @file   EntropyEncoderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "HMPIDWorkflow/EntropyEncoderSpec.h"
#include "HMPIDReconstruction/CTFCoder.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <TStopwatch.h>

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec(bool selIR = false);
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::hmpid::CTFCoder mCTFCoder;
  bool mSelIR = false;
  TStopwatch mTimer;
};

EntropyEncoderSpec::EntropyEncoderSpec(bool selIR) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder), mSelIR(selIR)
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
  auto triggers = pc.inputs().get<gsl::span<Trigger>>("triggers");
  auto digits = pc.inputs().get<gsl::span<Digit>>("digits");
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"HMP", "CTFDATA", 0, Lifetime::Timeframe});
  auto iosize = mCTFCoder.encode(buffer, triggers, digits);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "HMPID Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(bool selIR)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("triggers", "HMP", "INTRECORDS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digits", "HMP", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "HMP", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("HMP/Calib/CTFDictionary"));
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "hmpid-entropy-encoder",
    inputs,
    Outputs{{"HMP", "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, "HMP", "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace hmpid
} // namespace o2
