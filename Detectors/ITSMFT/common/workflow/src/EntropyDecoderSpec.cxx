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
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/PhysTrigger.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

template <int N>
std::string EntropyDecoderSpec<N>::getBinding(const std::string& name, int spec)
{
  return fmt::format("{}_{}", name, spec);
}

template <int N>
EntropyDecoderSpec<N>::EntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, const std::string& ctfdictOpt)
  : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Decoder, doStag, ctfdictOpt), mDoStaggering(doStag), mGetDigits(getDigits)
{
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
  mCTFCoder.setDictBinding(std::string("ctfdict_") + ID.getName());
}

template <int N>
void EntropyDecoderSpec<N>::init(o2::framework::InitContext& ic)
{
  mCTFCoder.template init<CTF>(ic);
  mMaskNoise = ic.options().get<bool>("mask-noise");
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
}

template <int N>
void EntropyDecoderSpec<N>::run(ProcessingContext& pc)
{
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) {
    mTimer.Reset();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  o2::ctf::CTFIOSize iosize;
  size_t ndigcl = 0, nrofs = 0;
  updateTimeDependentParams(pc);
  std::string nm = ID.getName();
  uint32_t nLayers = mDoStaggering ? DPLAlpideParam<N>::getNLayers() : 1;
  for (uint32_t iLayer = 0; iLayer < nLayers; iLayer++) {
    auto buff = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(getBinding(nm + "CTF", iLayer));
    // since the buff is const, we cannot use EncodedBlocks::relocate directly, instead we wrap its data to another flat object
    // const auto ctfImage = o2::itsmft::CTF::getImage(buff.data());
    const auto& ctf = o2::itsmft::CTF::getImage(buff.data());
    if (ctf.getHeader().maxStreams != nLayers) {
      LOGP(fatal, "Number of streams {} in the CTF header is not equal to NLayers {} from AlpideParam in {}staggered mode",
           ctf.getHeader().maxStreams, nLayers, mDoStaggering ? "" : "non-");
    }
    // this produces weird memory problems in unrelated devices, to be understood
    // auto& trigs = pc.outputs().make<std::vector<o2::itsmft::PhysTrigger>>(OutputRef{"phystrig"}); // dummy output
    auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(OutputRef{nm + "ROframes", iLayer});
    if (mGetDigits) {
      auto& digits = pc.outputs().make<std::vector<o2::itsmft::Digit>>(OutputRef{nm + "Digits", iLayer});
      if (buff.size()) {
        iosize += mCTFCoder.decode(ctf, rofs, digits, mNoiseMap, mPattIdConverter);
      }
      ndigcl += digits.size();
      nrofs += rofs.size();
    } else {
      auto& compcl = pc.outputs().make<std::vector<o2::itsmft::CompClusterExt>>(OutputRef{nm + "compClusters", iLayer});
      auto& patterns = pc.outputs().make<std::vector<unsigned char>>(OutputRef{nm + "patterns", iLayer});
      if (buff.size()) {
        iosize += mCTFCoder.decode(ctf, rofs, compcl, patterns, mNoiseMap, mPattIdConverter);
      }
      ndigcl += compcl.size();
    }
  }
  pc.outputs().snapshot({nm + "ctfrep", 0}, iosize);
  mTimer.Stop();
  LOGP(info, "Decoded {} {} in {} ROFs of {} streams ({}) in {}staggerd mode in {} s", ndigcl, mGetDigits ? "digits" : "clusters",
       nrofs, nLayers, iosize.asString(), mDoStaggering ? "" : "non-", mTimer.CpuTime() - cput);
}

template <int N>
void EntropyDecoderSpec<N>::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "{} Entropy Decoding total timing: Cpu: {:.3e} Real: {:.3e} s in {} slots",
       Origin.as<std::string>(), mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

template <int N>
void EntropyDecoderSpec<N>::updateTimeDependentParams(ProcessingContext& pc)
{
  std::string nm = ID.getName();
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // this params need to be queried only once
    if (mMaskNoise) {
      pc.inputs().get<o2::itsmft::NoiseMap*>(nm + "noise");
    }
    if (mGetDigits || mMaskNoise) {
      pc.inputs().get<o2::itsmft::TopologyDictionary*>(nm + "cldict");
    }
  }
  pc.inputs().get<o2::itsmft::DPLAlpideParam<N>*>(nm + "alppar");
  mCTFCoder.updateTimeDependentParams(pc, true);
}

template <int N>
void EntropyDecoderSpec<N>::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher(Origin, "NOISEMAP", 0)) {
    mNoiseMap = (o2::itsmft::NoiseMap*)obj;
    LOG(info) << Origin.as<std::string>() << " noise map updated";
    return;
  }
  if (matcher == ConcreteDataMatcher(Origin, "CLUSDICT", 0)) {
    LOG(info) << Origin.as<std::string>() << " cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    mPattIdConverter.setDictionary((const TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher(Origin, "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    return;
  }
  if (mCTFCoder.template finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

template <int N>
DataProcessorSpec getEntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, unsigned int sspec, const std::string& ctfdictOpt)
{
  constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  uint32_t nLayers = doStag ? DPLAlpideParam<N>::getNLayers() : 1;

  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  // this produces weird memory problems in unrelated devices, to be understood
  // outputs.emplace_back(OutputSpec{{"phystrig"}, Origin, "PHYSTRIG", 0, Lifetime::Timeframe});
  std::string nm = ID.getName();
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    if (getDigits) {
      outputs.emplace_back(OutputSpec{{nm + "Digits"}, Origin, "DIGITS", iLayer, Lifetime::Timeframe});
      outputs.emplace_back(OutputSpec{{nm + "ROframes"}, Origin, "DIGITSROF", iLayer, Lifetime::Timeframe});
    } else {
      outputs.emplace_back(OutputSpec{{nm + "compClusters"}, Origin, "COMPCLUSTERS", iLayer, Lifetime::Timeframe});
      outputs.emplace_back(OutputSpec{{nm + "ROframes"}, Origin, "CLUSTERSROF", iLayer, Lifetime::Timeframe});
      outputs.emplace_back(OutputSpec{{nm + "patterns"}, Origin, "PATTERNS", iLayer, Lifetime::Timeframe});
    }
    inputs.emplace_back(EntropyDecoderSpec<N>::getBinding(nm + "CTF", iLayer), Origin, "CTFDATA", sspec * 100 + iLayer, Lifetime::Timeframe);
  }
  outputs.emplace_back(OutputSpec{{nm + "ctfrep"}, Origin, "CTFDECREP", 0, Lifetime::Timeframe});

  inputs.emplace_back(nm + "alppar", Origin, "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Config/AlpideParam", Origin.as<std::string>())));
  inputs.emplace_back(nm + "noise", Origin, "NOISEMAP", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/NoiseMap", Origin.as<std::string>())));
  inputs.emplace_back(nm + "cldict", Origin, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/ClusterDictionary", Origin.as<std::string>())));
  if (ctfdictOpt.empty() || ctfdictOpt == "ccdb") {
    inputs.emplace_back(std::string{"ctfdict_"} + ID.getName(), Origin, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/CTFDictionaryTree", Origin.as<std::string>())));
  }
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, Lifetime::Condition, ccdbParamSpec("CTP/Config/TriggerOffsets"));

  return DataProcessorSpec{
    Origin == o2::header::gDataOriginITS ? "its-entropy-decoder" : "mft-entropy-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec<N>>(verbosity, doStag, getDigits, ctfdictOpt)},
    Options{{"mask-noise", VariantType::Bool, false, {"apply noise mask to digits or clusters (involves reclusterization)"}},
            {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

framework::DataProcessorSpec getITSEntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, unsigned int sspec, const std::string& ctfdictOpt)
{
  return getEntropyDecoderSpec<o2::detectors::DetID::ITS>(verbosity, doStag, getDigits, sspec, ctfdictOpt);
}

framework::DataProcessorSpec getMFTEntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, unsigned int sspec, const std::string& ctfdictOpt)
{
  return getEntropyDecoderSpec<o2::detectors::DetID::MFT>(verbosity, doStag, getDigits, sspec, ctfdictOpt);
}

} // namespace itsmft
} // namespace o2
