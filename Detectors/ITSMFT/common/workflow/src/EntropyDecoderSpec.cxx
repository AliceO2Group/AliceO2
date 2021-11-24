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
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITSMFTWorkflow/EntropyDecoderSpec.h"
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

EntropyDecoderSpec::EntropyDecoderSpec(o2::header::DataOrigin orig, int verbosity, bool getDigits)
  : mOrigin(orig), mCTFCoder(orig == o2::header::gDataOriginITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT), mGetDigits(getDigits)
{
  assert(orig == o2::header::gDataOriginITS || orig == o2::header::gDataOriginMFT);
  mTimer.Stop();
  mTimer.Reset();
  mCTFCoder.setVerbosity(verbosity);
}

void EntropyDecoderSpec::init(o2::framework::InitContext& ic)
{
  auto detID = mOrigin == o2::header::gDataOriginITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT;
  mCTFDictPath = ic.options().get<std::string>("ctf-dict");
  mClusDictPath = o2::header::gDataOriginITS ? o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().dictFilePath : o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance().dictFilePath;
  mClusDictPath = o2::base::NameConf::getAlpideClusterDictionaryFileName(detID, mClusDictPath);
  mMaskNoise = ic.options().get<bool>("mask-noise");
  mNoiseFilePath = o2::header::gDataOriginITS ? o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().noiseFilePath : o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance().noiseFilePath;
  mNoiseFilePath = o2::base::NameConf::getNoiseFileName(detID, mNoiseFilePath, "root");
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
      mCTFCoder.decode(o2::itsmft::CTF::getImage(buff.data()), rofs, digits, mNoiseMap.get(), mPattIdConverter);
    }
    mTimer.Stop();
    LOG(info) << "Decoded " << digits.size() << " digits in " << rofs.size() << " RO frames in " << mTimer.CpuTime() - cput << " s";
  } else {
    auto& compcl = pc.outputs().make<std::vector<o2::itsmft::CompClusterExt>>(OutputRef{"compClusters"});
    auto& patterns = pc.outputs().make<std::vector<unsigned char>>(OutputRef{"patterns"});
    if (buff.size()) {
      mCTFCoder.decode(o2::itsmft::CTF::getImage(buff.data()), rofs, compcl, patterns, mNoiseMap.get(), mPattIdConverter);
    }
    mTimer.Stop();
    LOG(info) << "Decoded " << compcl.size() << " clusters in " << rofs.size() << " RO frames in " << mTimer.CpuTime() - cput << " s";
  }
}

void EntropyDecoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "%s Entropy Decoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mOrigin.as<std::string>(), mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void EntropyDecoderSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool coderUpdated = false; // with dicts loaded from CCDB one should check also the validity of current object
  if (!coderUpdated) {
    coderUpdated = true;
    if (!mCTFDictPath.empty() && mCTFDictPath != "none") {
      mCTFCoder.createCoders(mCTFDictPath, o2::ctf::CTFCoderBase::OpType::Decoder);
    }

    if (mMaskNoise) {
      if (o2::utils::Str::pathExists(mNoiseFilePath)) {
        TFile* f = TFile::Open(mNoiseFilePath.data(), "old");
        mNoiseMap.reset((NoiseMap*)f->Get("ccdb_object"));
        LOG(info) << "Loaded noise map from " << mNoiseFilePath;
      }
      if (!mNoiseMap) {
        throw std::runtime_error("Noise masking was requested but noise mask was not provided");
      }
    }

    if (mGetDigits || mMaskNoise) {
      if (o2::utils::Str::pathExists(mClusDictPath)) {
        mPattIdConverter.loadDictionary(mClusDictPath);
        LOG(info) << "Loaded cluster topology dictionary from " << mClusDictPath;
      } else {
        LOG(info) << "Cluster topology dictionary is absent, all cluster patterns expected to be stored explicitly";
      }
    }
  }
}

DataProcessorSpec getEntropyDecoderSpec(o2::header::DataOrigin orig, int verbosity, bool getDigits)
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

  return DataProcessorSpec{
    EntropyDecoderSpec::getName(orig),
    Inputs{InputSpec{"ctf", orig, "CTFDATA", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<EntropyDecoderSpec>(orig, verbosity, getDigits)},
    Options{
      {"ctf-dict", VariantType::String, o2::base::NameConf::getCTFDictFileName(), {"File of CTF decoding dictionary"}},
      {"mask-noise", VariantType::Bool, false, {"apply noise mask to digits or clusters (involves reclusterization)"}}}};
}

} // namespace itsmft
} // namespace o2
