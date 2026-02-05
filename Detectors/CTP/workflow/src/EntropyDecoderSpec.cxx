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

/// @file   EntropyDecoderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "CTPWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ctp
{
EntropyDecoderSpec::EntropyDecoderSpec(int verbosity, const std::string& ctfdictOpt) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder, ctfdictOpt)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setSupportBCShifts(true);
  mCTFCoder.setDictBinding("ctfdict_CTP");
}

void EntropyDecoderSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
  bool decodeinps = !ic.options().get<bool>("ignore-ctpinputs-decoding-ctf");
  mCTFCoder.setDecodeInps(decodeinps);
  LOG(info) << "Decode inputs:" << decodeinps;
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  o2::ctf::CTFIOSize iosize;

  updateTimeDependentParams(pc);
  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf_CTP");
  auto& digits = pc.outputs().make<std::vector<CTPDigit>>(OutputRef{"digits"});
  auto& lumi = pc.outputs().make<LumiInfo>(OutputRef{"CTPLumi"});

  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  if (buff.size()) {
    const auto ctfImage = o2::ctp::CTF::getImage(buff.data());
    iosize = mCTFCoder.decode(ctfImage, digits, lumi);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  LOG(info) << "Decoded " << digits.size() << " CTP digits, (" << iosize.asString() << ") in " << mTimer.CpuTime() - cput << " s";
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "CTP Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}
void EntropyDecoderSpec::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  mCTFCoder.updateTimeDependentParams(pc, true);
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) {
    const auto ctpcfg = pc.inputs().get<o2::ctp::CTPConfiguration*>("ctpconfig");
    if (mCTFCoder.getDecodeInps()) {
      const auto ctpcfg = pc.inputs().get<o2::ctp::CTPConfiguration*>("ctpconfig");
      if (ctpcfg != nullptr) {
        mCTFCoder.setCTPConfig(*ctpcfg);
        LOG(info) << "ctpconfig for run done:" << mCTFCoder.getCTPConfig().getRunNumber();
      }
    }
  }
}

DataProcessorSpec getEntropyDecoderSpec(int verbosity, unsigned int sspec, const std::string& ctfdictOpt)
{
  std::vector<OutputSpec> outputs{
    OutputSpec{{"digits"}, "CTP", "DIGITS", 0, Lifetime::Timeframe},
    OutputSpec{{"CTPLumi"}, "CTP", "LUMI", 0, Lifetime::Timeframe},
    OutputSpec{{"ctfrep"}, "CTP", "CTFDECREP", 0, Lifetime::Timeframe}};

  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf_CTP", "CTP", "CTFDATA", sspec, Lifetime::Timeframe);

  if (ctfdictOpt.empty() || ctfdictOpt == "ccdb") {
    inputs.emplace_back("ctfdict_CTP", "CTP", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/CTFDictionaryTree"));
  }
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));
  inputs.emplace_back("ctpconfig", "CTP", "CTPCONFIG", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/Config", 1));
  return DataProcessorSpec{
    "ctp-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(verbosity, ctfdictOpt)},
    Options{{"ignore-ctpinputs-decoding-ctf", VariantType::Bool, false, {"Inputs alignment: false - CTF decoder - has to be compatible with reco: allowed options: 10,01,00"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}
} // namespace ctp
} // namespace o2
