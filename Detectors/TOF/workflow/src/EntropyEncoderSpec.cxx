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
#include "TOFBase/Digit.h"
#include "TOFWorkflowUtils/EntropyEncoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

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
  auto compDigits = pc.inputs().get<gsl::span<Digit>>("compDigits");
  auto pspan = pc.inputs().get<gsl::span<uint8_t>>("patterns");
  auto rofs = pc.inputs().get<gsl::span<ReadoutWindowData>>("ROframes");
  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{o2::header::gDataOriginTOF, "CTFDATA", 0, Lifetime::Timeframe});
  auto iosize = mCTFCoder.encode(buffer, rofs, compDigits, pspan);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "TOF Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(bool selIR)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compDigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "TOF", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/CTFDictionary"));
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "tof-entropy-encoder",
    inputs,
    Outputs{{o2::header::gDataOriginTOF, "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, "TOF", "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"irframe-shift", VariantType::Int, -o2::tof::Geo::LATENCYWINDOW_IN_BC, {"IRFrame shift to account for latency"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace tof
} // namespace o2
