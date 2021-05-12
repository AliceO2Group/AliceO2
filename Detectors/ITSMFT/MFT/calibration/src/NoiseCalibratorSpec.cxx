// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorSpec.cxx

#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "MFTCalibration/NoiseCalibratorSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "FairLogger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;

// Remove leading whitespace
std::string ltrimSpace(std::string src)
{
  return src.erase(0, src.find_first_not_of(' '));
}

// Remove trailing whitespace
std::string rtrimSpace(std::string src)
{
  return src.erase(src.find_last_not_of(' ') + 1);
}

// Remove leading/trailing whitespace
std::string trimSpace(std::string const& src)
{
  return ltrimSpace(rtrimSpace(src));
}

std::vector<std::string> splitString(const std::string& src, char delim, bool trim = false)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    token = (trim ? trimSpace(token) : token);
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}

namespace o2
{
namespace mft
{

NoiseCalibratorSpec::NoiseCalibratorSpec(bool useDigits)
  : mDigits(useDigits)
{
}

void NoiseCalibratorSpec::init(InitContext& ic)
{
  auto onepix = ic.options().get<bool>("1pix-only");
  LOG(INFO) << "Fast 1=pixel calibration: " << onepix;
  auto probT = ic.options().get<float>("prob-threshold");
  LOG(INFO) << "Setting the probability threshold to " << probT;
  auto HBperTF = ic.options().get<int>("hb-per-tf");
  LOG(INFO) << "Nb of HBF per TF used : " << HBperTF;

  mPath = ic.options().get<std::string>("path");
  mMeta = ic.options().get<std::string>("meta");
  mStart = ic.options().get<int64_t>("tstart");
  mEnd = ic.options().get<int64_t>("tend");

  mCalibrator = std::make_unique<CALIBRATOR>(onepix, probT, HBperTF);
}

void NoiseCalibratorSpec::run(ProcessingContext& pc)
{
  if (mDigits) { 
    const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");

    if (mCalibrator->processTimeFrame(digits, rofs)) {
      LOG(INFO) << "Minimum number of noise counts has been reached !";
      sendOutput(pc.outputs());
      pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
    }
  }else{
    const auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
    gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

    if (mCalibrator->processTimeFrame(compClusters, patterns, rofs)) {
      LOG(INFO) << "Minimum number of noise counts has been reached !";
      sendOutput(pc.outputs());
      pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
    }
  }
}

void NoiseCalibratorSpec::sendOutput(DataAllocator& output)
{
  mCalibrator->finalize();

  long tstart = mStart;
  if (tstart == -1) {
    tstart = o2::ccdb::getCurrentTimestamp();
  }
  long tend = mEnd;
  if (tend == -1) {
    constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
    tend = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);
  }

  auto toKeyValPairs = [](std::vector<std::string> const& tokens) {
    std::vector<std::pair<std::string, std::string>> pairs;

    for (auto& token : tokens) {
      auto keyval = splitString(token, '=');
      if (keyval.size() != 2) {
        // LOG(FATAL) << "Illegal command-line key/value string: " << token;
        continue;
      }

      std::pair<std::string, std::string> pair = std::make_pair(keyval[0], trimSpace(keyval[1]));
      pairs.push_back(pair);
    }

    return pairs;
  };
  std::map<std::string, std::string> meta;
  auto keyvalues = toKeyValPairs(splitString(mMeta, ';', true));

  // fill meta map
  for (auto& p : keyvalues) {
    meta[p.first] = p.second;
  }

  long startTF, endTF;

#ifdef TIME_SLOT_CALIBRATION
  const auto& payload = mCalibrator->getNoiseMap(startTF, endTF);
#else
  const auto& payload = mCalibrator->getNoiseMap();
#endif

  o2::ccdb::CcdbObjectInfo info(mPath, "NoiseMap", "noise.root", meta, tstart, tend);
  auto flName = o2::ccdb::CcdbApi::generateFileName("noise");
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  info.setFileName(flName);
  LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  using clbUtils = o2::calibration::Utils;
  output.snapshot(
    Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 0}, *image.get());
  output.snapshot(
    Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 0}, info);
}

void NoiseCalibratorSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  sendOutput(ec.outputs());
}

DataProcessorSpec getNoiseCalibratorSpec(bool useDigits)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginMFT;
  std::vector<InputSpec> inputs;
  if (useDigits) {
    inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  }else{
    inputs.emplace_back("compClusters", detOrig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", detOrig, "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", detOrig, "CLUSTERSROF", 0, Lifetime::Timeframe);
  }

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});

  return DataProcessorSpec{
    "mft-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(useDigits)},
    Options{
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}},
      {"tstart", VariantType::Int64, -1ll, {"Start of validity timestamp"}},
      {"tend", VariantType::Int64, -1ll, {"End of validity timestamp"}},
      {"path", VariantType::String, "/MFT/Calib/NoiseMap", {"Path to write to in CCDB"}},
      {"meta", VariantType::String, "", {"meta data to write in CCDB"}},
      {"hb-per-tf", VariantType::Int, 256, {"Number of HBF per TF"}}}};
}

} // namespace mft
} // namespace o2
