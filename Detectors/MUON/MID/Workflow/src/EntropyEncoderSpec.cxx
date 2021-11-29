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
#include "Framework/DataRef.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace mid
{

EntropyEncoderSpec::EntropyEncoderSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyEncoderSpec::init(o2::framework::InitContext& ic)
{
  std::string dictPath = ic.options().get<std::string>("ctf-dict");
  mCTFCoder.setMemMarginFactor(ic.options().get<float>("mem-factor"));
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCodersFromFile<CTF>(dictPath, o2::ctf::CTFCoderBase::OpType::Encoder);
  }
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  CTFHelper::TFData tfData;
  std::vector<InputSpec>
    filter = {
      {"check", ConcreteDataTypeMatcher{header::gDataOriginMID, "DATA"}, Lifetime::Timeframe},
      {"check", ConcreteDataTypeMatcher{header::gDataOriginMID, "DATAROF"}, Lifetime::Timeframe},
    };
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), filter)) {
    auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
    if (dh->subSpecification >= NEvTypes) {
      throw std::runtime_error(fmt::format("SubSpecification={} does not match EvenTypes for {}", dh->subSpecification, dh->dataDescription.as<std::string>()));
    }
    if (DataRefUtils::match(inputRef, "cols")) {
      tfData.colData[dh->subSpecification] = pc.inputs().get<gsl::span<o2::mid::ColumnData>>(inputRef);
    }
    if (DataRefUtils::match(inputRef, "rofs")) {
      tfData.rofData[dh->subSpecification] = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>(inputRef);
    }
  }
  // build references for looping over the data in BC increasing direction
  tfData.buildReferences();

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{header::gDataOriginMID, "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, tfData);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  //  eeb->print();
  mTimer.Stop();
  LOG(info) << "Created encoded data of size " << eeb->size() << " for MID in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "MID Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("rofs", ConcreteDataTypeMatcher(header::gDataOriginMID, "DATAROF"), Lifetime::Timeframe);
  inputs.emplace_back("cols", ConcreteDataTypeMatcher(header::gDataOriginMID, "DATA"), Lifetime::Timeframe);

  return DataProcessorSpec{
    "mid-entropy-encoder",
    inputs,
    Outputs{{header::gDataOriginMID, "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF encoding dictionary"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace mid
} // namespace o2
