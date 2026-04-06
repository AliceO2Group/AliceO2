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
/// @brief  Convert MID DATA to CTF (EncodedBlocks)
/// @author ruben.shahoyan@cern.ch

#include "MIDWorkflow/EntropyEncoderSpec.h"

#include <vector>
#include <unordered_map>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace mid
{
EntropyEncoderSpec::EntropyEncoderSpec(bool selIR, const std::string& ctfdictOpt) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder, ctfdictOpt), mSelIR(selIR)
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
  mCTFCoder.updateTimeDependentParams(pc, true);
  CTFHelper::TFData tfData;
  size_t insize = 0;
  for (o2::header::DataHeader::SubSpecificationType subSpec = 0; subSpec < NEvTypes; ++subSpec) {
    tfData.colData[subSpec] = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(fmt::format("cols_{}", subSpec));
    insize += tfData.colData[subSpec].size() * sizeof(o2::mid::ColumnData);

    tfData.rofData[subSpec] = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(fmt::format("rofs_{}", subSpec));
    insize += tfData.rofData[subSpec].size() * sizeof(o2::mid::ROFRecord);
  }

  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  // build references for looping over the data in BC increasing direction
  tfData.buildReferences(mCTFCoder.getIRFramesSelector());

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{header::gDataOriginMID, "CTFDATA", 0});
  auto iosize = mCTFCoder.encode(buffer, tfData);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  iosize.rawIn = insize;
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "MID Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(bool selIR, const std::string& ctfdictOpt)
{
  std::vector<InputSpec> inputs;
  for (o2::header::DataHeader::SubSpecificationType subSpec = 0; subSpec < NEvTypes; ++subSpec) {
    inputs.emplace_back(fmt::format("cols_{}", subSpec), header::gDataOriginMID, "DATA", subSpec, Lifetime::Timeframe);
    inputs.emplace_back(fmt::format("rofs_{}", subSpec), header::gDataOriginMID, "DATAROF", subSpec, Lifetime::Timeframe);
  }

  if (ctfdictOpt.empty() || ctfdictOpt == "ccdb") {
    inputs.emplace_back("ctfdict", header::gDataOriginMID, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("MID/Calib/CTFDictionaryTree"));
  }
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "mid-entropy-encoder",
    inputs,
    Outputs{{header::gDataOriginMID, "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, header::gDataOriginMID, "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(selIR, ctfdictOpt)},
    Options{{"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}
} // namespace mid
} // namespace o2
