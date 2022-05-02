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
#include "EMCALWorkflow/EntropyEncoderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2
{
namespace emcal
{

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
  auto triggers = pc.inputs().get<gsl::span<TriggerRecord>>("triggers");
  auto cells = pc.inputs().get<gsl::span<Cell>>("cells");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"EMC", "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, triggers, cells);
  auto sz = mCTFCoder.finaliseCTFOutput<CTF>(buffer);
  mTimer.Stop();
  LOG(info) << "Created encoded data of size " << sz << " for EMCAL in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "EMCAL Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("triggers", "EMC", "CELLSTRGR", 0, Lifetime::Timeframe);
  inputs.emplace_back("cells", "EMC", "CELLS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "EMC", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("EMC/Calib/CTFDictionary"));

  return DataProcessorSpec{
    "emcal-entropy-encoder",
    inputs,
    Outputs{{"EMC", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace emcal
} // namespace o2
