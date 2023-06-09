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
  mCTFCoder.updateTimeDependentParams(pc, true);
  CTFHelper::TFData tfData;
  std::vector<InputSpec>
    filter = {
      {"check", ConcreteDataTypeMatcher{header::gDataOriginMID, "DATA"}, Lifetime::Timeframe},
      {"check", ConcreteDataTypeMatcher{header::gDataOriginMID, "DATAROF"}, Lifetime::Timeframe},
    };
  size_t insize = 0;
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), filter)) {
    auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
    if (dh->subSpecification >= NEvTypes) {
      throw std::runtime_error(fmt::format("SubSpecification={} does not match EvenTypes for {}", dh->subSpecification, dh->dataDescription.as<std::string>()));
    }
    if (DataRefUtils::match(inputRef, "cols")) {
      tfData.colData[dh->subSpecification] = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(inputRef);
      insize += tfData.colData[dh->subSpecification].size() * sizeof(o2::mid::ColumnData);
    }
    if (DataRefUtils::match(inputRef, "rofs")) {
      tfData.rofData[dh->subSpecification] = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(inputRef);
      insize += tfData.rofData[dh->subSpecification].size() * sizeof(o2::mid::ROFRecord);
    }
  }
  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  // build references for looping over the data in BC increasing direction
  tfData.buildReferences(mCTFCoder.getIRFramesSelector());

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{header::gDataOriginMID, "CTFDATA", 0, Lifetime::Timeframe});
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

DataProcessorSpec getEntropyEncoderSpec(bool selIR)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("rofs", ConcreteDataTypeMatcher(header::gDataOriginMID, "DATAROF"), Lifetime::Timeframe);
  inputs.emplace_back("cols", ConcreteDataTypeMatcher(header::gDataOriginMID, "DATA"), Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", header::gDataOriginMID, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("MID/Calib/CTFDictionaryTree"));
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "mid-entropy-encoder",
    inputs,
    Outputs{{header::gDataOriginMID, "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, header::gDataOriginMID, "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

} // namespace mid
} // namespace o2
