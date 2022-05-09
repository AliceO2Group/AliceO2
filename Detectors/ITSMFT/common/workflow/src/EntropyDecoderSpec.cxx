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
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITSMFTWorkflow/EntropyDecoderSpec.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

EntropyDecoderSpec::EntropyDecoderSpec(o2::header::DataOrigin orig, int verbosity, bool getDigits)
  : mOrigin(orig), mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder, orig == o2::header::gDataOriginITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT), mGetDigits(getDigits)
{
  assert(orig == o2::header::gDataOriginITS || orig == o2::header::gDataOriginMFT);
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
  mMaskNoise = ic.options().get<bool>("mask-noise");
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
}

void EntropyDecoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  updateTimeDependentParams(pc);

  auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>("ctf");
  // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
  //  const auto ctfImage = o2::itsmft::CTF::getImage(buff.data());

  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(OutputRef{"ROframes"});
  if (mGetDigits) {
    auto& digits = pc.outputs().make<std::vector<o2::itsmft::Digit>>(OutputRef{"Digits"});
    if (buff.size()) {
      mCTFCoder.decode(o2::itsmft::CTF::getImage(buff.data()), rofs, digits, mNoiseMap, mPattIdConverter);
    }
    mTimer.Stop();
    LOG(info) << "Decoded " << digits.size() << " digits in " << rofs.size() << " RO frames in " << mTimer.CpuTime() - cput << " s";
  } else {
    auto& compcl = pc.outputs().make<std::vector<o2::itsmft::CompClusterExt>>(OutputRef{"compClusters"});
    auto& patterns = pc.outputs().make<std::vector<unsigned char>>(OutputRef{"patterns"});
    if (buff.size()) {
      mCTFCoder.decode(o2::itsmft::CTF::getImage(buff.data()), rofs, compcl, patterns, mNoiseMap, mPattIdConverter);
    }
    mTimer.Stop();
    LOG(info) << "Decoded " << compcl.size() << " clusters in " << rofs.size() << " RO frames in " << mTimer.CpuTime() - cput << " s";
  }
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "%s Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mOrigin.as<std::string>(), mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void EntropyDecoderSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  if (mMaskNoise) {
    pc.inputs().get<o2::itsmft::NoiseMap*>("noise").get();
  }
  if (mGetDigits || mMaskNoise) {
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict");
  }
  mCTFCoder.updateTimeDependentParams(pc);
}

void EntropyDecoderSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher(mOrigin, "NOISEMAP", 0)) {
    mNoiseMap = (o2::itsmft::NoiseMap*)obj;
    LOG(info) << mOrigin.as<std::string>() << " noise map updated";
    return;
  }
  if (matcher == ConcreteDataMatcher(mOrigin, "CLUSDICT", 0)) {
    LOG(info) << mOrigin.as<std::string>() << " cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    mPattIdConverter.setDictionary((const TopologyDictionary*)obj);
    return;
  }
  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getEntropyDecoderSpec(o2::header::DataOrigin orig, int verbosity, bool getDigits, unsigned int sspec)
{
  std::vector<OutputSpec> outputs;
  if (getDigits) {
    outputs.emplace_back(OutputSpec{{"Digits"}, orig, "DIGITS", 0, Lifetime::Timeframe});
    outputs.emplace_back(OutputSpec{{"ROframes"}, orig, "DIGITSROF", 0, Lifetime::Timeframe});
  } else {
    outputs.emplace_back(OutputSpec{{"compClusters"}, orig, "COMPCLUSTERS", 0, Lifetime::Timeframe});
    outputs.emplace_back(OutputSpec{{"ROframes"}, orig, "CLUSTERSROF", 0, Lifetime::Timeframe});
    outputs.emplace_back(OutputSpec{{"patterns"}, orig, "PATTERNS", 0, Lifetime::Timeframe});
  }
  std::vector<InputSpec> inputs;
  inputs.emplace_back("ctf", orig, "CTFDATA", sspec, Lifetime::Timeframe);
  inputs.emplace_back("noise", orig, "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/NoiseMap", orig.as<std::string>())));
  inputs.emplace_back("cldict", orig, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/ClusterDictionary", orig.as<std::string>())));
  inputs.emplace_back("ctfdict", orig, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/CTFDictionary", orig.as<std::string>())));

  return DataProcessorSpec{
    EntropyDecoderSpec::getName(orig),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(orig, verbosity, getDigits)},
    Options{
      {"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
      {"mask-noise", VariantType::Bool, false, {"apply noise mask to digits or clusters (involves reclusterization)"}},
      {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}}}};
}

} // namespace itsmft
} // namespace o2
