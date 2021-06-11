// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EntropyEncoderSpec.cxx
/// @brief  Convert MID DATA to CTF (EncodedBlocks)
/// @author ruben.shahoyan@cern.ch

#include "MIDWorkflow/EntropyEncoderSpec.h"

#include <vector>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
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
  if (!dictPath.empty() && dictPath != "none") {
    mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Encoder);
  }
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto rofs = pc.inputs().get<gsl::span<o2::mid::ROFRecord>>("rofs");
  auto cols = pc.inputs().get<gsl::span<o2::mid::ColumnData>>("cols");

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"MID", "CTFDATA", 0, Lifetime::Timeframe});
  mCTFCoder.encode(buffer, rofs, cols);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  //  eeb->print();
  mTimer.Stop();
  LOG(INFO) << "Created encoded data of size " << eeb->size() << " for MID in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "MID Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("rofs", "MID", "DATAROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cols", "MID", "DATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "mid-entropy-encoder",
    inputs,
    Outputs{{"MID", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>()},
    Options{{"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF encoding dictionary"}}}};
}

} // namespace mid
} // namespace o2
