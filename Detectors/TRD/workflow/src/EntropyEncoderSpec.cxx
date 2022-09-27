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
#include "TRDWorkflow/EntropyEncoderSpec.h"
#include "TRDReconstruction/CTFCoder.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <TStopwatch.h>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec(bool selIR);
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::trd::CTFCoder mCTFCoder;
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
  int checkBogus = ic.options().get<int>("bogus-trigger-check");
  mCTFCoder.setCheckBogusTrig(checkBogus < 0 ? 0x7fffffff : checkBogus);
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  mCTFCoder.updateTimeDependentParams(pc);
  auto triggers = pc.inputs().get<gsl::span<TriggerRecord>>("triggers");
  auto tracklets = pc.inputs().get<gsl::span<Tracklet64>>("tracklets");
  auto digits = pc.inputs().get<gsl::span<Digit>>("digits");
  mCTFCoder.setFirstTFOrbit(pc.services().get<o2::framework::TimingInfo>().firstTForbit);
  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"TRD", "CTFDATA", 0, Lifetime::Timeframe});
  auto iosize = mCTFCoder.encode(buffer, triggers, tracklets, digits);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(bool selIR)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("triggers", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracklets", "TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digits", "TRD", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "TRD", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CTFDictionary"));
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "trd-entropy-encoder",
    inputs,
    Outputs{{"TRD", "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, "TRD", "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}},
            {"bogus-trigger-check", VariantType::Int, 10, {"max bogus triggers to report, all if < 0"}}}};
}

} // namespace trd
} // namespace o2
