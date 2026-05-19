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
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITSMFTWorkflow/EntropyEncoderSpec.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

template <int N>
std::string EntropyEncoderSpec<N>::getBinding(const std::string& name, int spec)
{
  return fmt::format("{}_{}", name, spec);
}

template <int N>
EntropyEncoderSpec<N>::EntropyEncoderSpec(bool doStag, bool selIR, const std::string& ctfdictOpt)
  : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder, doStag, ctfdictOpt),
    mSelIR(selIR),
    mDoStaggering(doStag)
{
  mTimer.Stop();
  mTimer.Reset();
}

template <int N>
void EntropyEncoderSpec<N>::init(o2::framework::InitContext& ic)
{
  mCTFCoder.template init<CTF>(ic);
}

template <int N>
void EntropyEncoderSpec<N>::run(ProcessingContext& pc)
{
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) {
    mTimer.Reset();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  updateTimeDependentParams(pc);

  uint32_t nLayers = mDoStaggering ? DPLAlpideParam<N>::getNLayers() : 1;

  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  o2::ctf::CTFIOSize iosize{};
  for (uint32_t iLayer = 0; iLayer < nLayers; iLayer++) {
    auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>(getBinding("compClusters", iLayer));
    auto pspan = pc.inputs().get<gsl::span<unsigned char>>(getBinding("patterns", iLayer));
    auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>(getBinding("ROframes", iLayer));

    auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{Origin, "CTFDATA", iLayer});
    iosize += mCTFCoder.encode(buffer, rofs, compClusters, pspan, mPattIdConverter, iLayer);
  }
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

template <int N>
void EntropyEncoderSpec<N>::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "{} Entropy Encoding total timing: Cpu: {:.3e} Real: {:.3e} s in {} slots",
       Origin.as<std::string>(), mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

template <int N>
void EntropyEncoderSpec<N>::updateTimeDependentParams(ProcessingContext& pc)
{
  mCTFCoder.updateTimeDependentParams(pc, true);
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // this params need to be queried only once
    if (mSelIR) {
      pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict");
    }
  }
  pc.inputs().get<o2::itsmft::DPLAlpideParam<N>*>("alppar");
}

template <int N>
void EntropyEncoderSpec<N>::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher(Origin, "CLUSDICT", 0)) {
    LOG(info) << Origin.as<std::string>() << " cluster dictionary updated";
    mPattIdConverter.setDictionary((const TopologyDictionary*)obj);
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher(Origin, "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    return;
  }

  if (mCTFCoder.template finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

template <int N>
DataProcessorSpec getEntropyEncoderSpec(bool doStag, bool selIR, const std::string& ctfdictOpt)
{
  constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  const auto& par = DPLAlpideParam<N>::Instance();
  uint32_t nLayers = doStag ? DPLAlpideParam<N>::getNLayers() : 1;

  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    inputs.emplace_back(EntropyEncoderSpec<N>::getBinding("compClusters", iLayer), Origin, "COMPCLUSTERS", iLayer, Lifetime::Timeframe);
    inputs.emplace_back(EntropyEncoderSpec<N>::getBinding("patterns", iLayer), Origin, "PATTERNS", iLayer, Lifetime::Timeframe);
    inputs.emplace_back(EntropyEncoderSpec<N>::getBinding("ROframes", iLayer), Origin, "CLUSTERSROF", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(Origin, "CTFDATA", iLayer, Lifetime::Timeframe);
  }
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
    inputs.emplace_back("cldict", Origin, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/ClusterDictionary", Origin.as<std::string>())));
  }
  inputs.emplace_back("alppar", Origin, "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Config/AlpideParam", Origin.as<std::string>())));

  if (ctfdictOpt.empty() || ctfdictOpt == "ccdb") {
    inputs.emplace_back("ctfdict", Origin, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/CTFDictionaryTree", Origin.as<std::string>())));
  }
  outputs.emplace_back(OutputSpec{{"ctfrep"}, Origin, "CTFENCREP", 0, Lifetime::Timeframe});
  return DataProcessorSpec{
    Origin == o2::header::gDataOriginITS ? "its-entropy-encoder" : "mft-entropy-encoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec<N>>(doStag, selIR, ctfdictOpt)},
    Options{{"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}},
            {"ans-version", VariantType::String, {"version of ans entropy coder implementation to use"}}}};
}

framework::DataProcessorSpec getITSEntropyEncoderSpec(bool doStag, bool selIR, const std::string& ctfdictOpt)
{
  return getEntropyEncoderSpec<o2::detectors::DetID::ITS>(doStag, selIR, ctfdictOpt);
}

framework::DataProcessorSpec getMFTEntropyEncoderSpec(bool doStag, bool selIR, const std::string& ctfdictOpt)
{
  return getEntropyEncoderSpec<o2::detectors::DetID::MFT>(doStag, selIR, ctfdictOpt);
}

} // namespace itsmft
} // namespace o2
