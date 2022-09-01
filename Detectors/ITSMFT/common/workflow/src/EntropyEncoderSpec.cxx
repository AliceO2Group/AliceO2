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
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

EntropyEncoderSpec::EntropyEncoderSpec(o2::header::DataOrigin orig, bool selIR)
  : mOrigin(orig), mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder, orig == o2::header::gDataOriginITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT), mSelIR(selIR)
{
  assert(orig == o2::header::gDataOriginITS || orig == o2::header::gDataOriginMFT);
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyEncoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  updateTimeDependentParams(pc);

  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  auto pspan = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
  }
  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{mOrigin, "CTFDATA", 0, Lifetime::Timeframe});
  auto iosize = mCTFCoder.encode(buffer, rofs, compClusters, pspan, mPattIdConverter, mStrobeLength);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  mTimer.Stop();
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "%s Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mOrigin.as<std::string>(), mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void EntropyEncoderSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  mCTFCoder.updateTimeDependentParams(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    if (mSelIR) {
      pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict");
      if (mOrigin == o2::header::gDataOriginITS) {
        pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alppar");
      } else {
        pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>*>("alppar");
      }
    }
  }
}

void EntropyEncoderSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher(mOrigin, "CLUSDICT", 0)) {
    LOG(info) << mOrigin.as<std::string>() << " cluster dictionary updated";
    mPattIdConverter.setDictionary((const TopologyDictionary*)obj);
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher(mOrigin, "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    if (mOrigin == o2::header::gDataOriginITS) {
      const auto& par = DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
      mStrobeLength = par.roFrameLengthInBC;
    } else {
      const auto& par = DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
      mStrobeLength = par.roFrameLengthInBC;
    }
    return;
  }

  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getEntropyEncoderSpec(o2::header::DataOrigin orig, bool selIR)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", orig, "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", orig, "CLUSTERSROF", 0, Lifetime::Timeframe);
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
    inputs.emplace_back("cldict", orig, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/ClusterDictionary", orig.as<std::string>())));
    inputs.emplace_back("alppar", orig, "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Config/AlpideParam", orig.as<std::string>())));
  }
  inputs.emplace_back("ctfdict", orig, "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/CTFDictionary", orig.as<std::string>())));
  return DataProcessorSpec{
    orig == o2::header::gDataOriginITS ? "its-entropy-encoder" : "mft-entropy-encoder",
    inputs,
    Outputs{{orig, "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, orig, "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(orig, selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace itsmft
} // namespace o2
