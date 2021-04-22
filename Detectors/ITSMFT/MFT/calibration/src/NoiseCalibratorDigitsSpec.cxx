// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorDigitsSpec.cxx

#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "MFTCalibration/NoiseCalibratorDigitsSpec.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"

#include "FairLogger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

// Remove leading whitespace
std::string ltrimSpaceStream(std::string src)
{
  return src.erase(0, src.find_first_not_of(' '));
}

// Remove trailing whitespace
std::string rtrimSpaceStream(std::string src)
{
  return src.erase(src.find_last_not_of(' ') + 1);
}

// Remove leading/trailing whitespace
std::string trimSpaceStream(std::string const& src)
{
  return ltrimSpaceStream(rtrimSpaceStream(src));
}

std::vector<std::string> splitStringStream(const std::string& src, char delim, bool trim = false)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    token = (trim ? trimSpaceStream(token) : token);
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}
void NoiseCalibratorDigitsSpec::init(InitContext& ic)
{
  auto onepix = ic.options().get<bool>("1pix-only");
  LOG(INFO) << "Fast 1=pixel calibration: " << onepix;
  auto probT = ic.options().get<float>("prob-threshold");
  LOG(INFO) << "Setting the probability threshold to " << probT;

//  mPath = ic.options().get<std::string>("path");
  mMeta = ic.options().get<std::string>("meta");
  mStart = ic.options().get<int64_t>("tstart");
  mEnd = ic.options().get<int64_t>("tend");

  mCalibrator = std::make_unique<CALIBRATORDIGITS>(onepix, probT);
}

void NoiseCalibratorDigitsSpec::run(ProcessingContext& pc)
{
  const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");

  if (mCalibrator->processTimeFrame(digits, rofs)) {
    LOG(INFO) << "Minimum number of noise counts has been reached !";
    sendOutput(pc.outputs());
    pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
  }
}

void NoiseCalibratorDigitsSpec::sendOutput(DataAllocator& output)
{
  mCalibrator->finalize();

  const auto& payloadH0F0 = mCalibrator->getNoiseMapH0F0();
  const auto& payloadH0F1 = mCalibrator->getNoiseMapH0F1();
  const auto& payloadH1F0 = mCalibrator->getNoiseMapH1F0();
  const auto& payloadH1F1 = mCalibrator->getNoiseMapH1F1();

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
      auto keyval = splitStringStream(token, '=');
      if (keyval.size() != 2) {
        // LOG(FATAL) << "Illegal command-line key/value string: " << token;
        continue;
      }

      std::pair<std::string, std::string> pair = std::make_pair(keyval[0], trimSpaceStream(keyval[1]));
      pairs.push_back(pair);
    }

    return pairs;
  };
  std::map<std::string, std::string> meta;
  auto keyvalues = toKeyValPairs(splitStringStream(mMeta, ';', true));

  // fill meta map
  for (auto& p : keyvalues) {
    meta[p.first] = p.second;
  }

  o2::ccdb::CcdbObjectInfo infoH0F0(mCalibrator->getPathH0F0(), "NoiseMap", "noise.root", meta, tstart, tend);
  o2::ccdb::CcdbObjectInfo infoH0F1(mCalibrator->getPathH0F1(), "NoiseMap", "noise.root", meta, tstart, tend);
  o2::ccdb::CcdbObjectInfo infoH1F0(mCalibrator->getPathH1F0(), "NoiseMap", "noise.root", meta, tstart, tend);
  o2::ccdb::CcdbObjectInfo infoH1F1(mCalibrator->getPathH1F1(), "NoiseMap", "noise.root", meta, tstart, tend);
  auto flNameH0F0 = o2::ccdb::CcdbApi::generateFileName("noiseH0F0");
  auto flNameH0F1 = o2::ccdb::CcdbApi::generateFileName("noiseH0F1");
  auto flNameH1F0 = o2::ccdb::CcdbApi::generateFileName("noiseH1F0");
  auto flNameH1F1 = o2::ccdb::CcdbApi::generateFileName("noiseH1F1");
  auto imageH0F0 = o2::ccdb::CcdbApi::createObjectImage(&payloadH0F0, &infoH0F0);
  auto imageH0F1 = o2::ccdb::CcdbApi::createObjectImage(&payloadH0F1, &infoH0F1);
  auto imageH1F0 = o2::ccdb::CcdbApi::createObjectImage(&payloadH1F0, &infoH1F0);
  auto imageH1F1 = o2::ccdb::CcdbApi::createObjectImage(&payloadH1F1, &infoH1F1);
  infoH0F0.setFileName(flNameH0F0);
  infoH0F1.setFileName(flNameH0F1);
  infoH1F0.setFileName(flNameH1F0);
  infoH1F1.setFileName(flNameH1F1);
  LOG(INFO) << "Sending object " << infoH0F0.getPath() << "/" << infoH0F0.getFileName()
            << " of size " << imageH0F0->size()
            << " bytes, valid for " << infoH0F0.getStartValidityTimestamp()
            << " : " << infoH0F0.getEndValidityTimestamp();
  LOG(INFO) << "Sending object " << infoH0F1.getPath() << "/" << infoH0F1.getFileName()
            << " of size " << imageH0F1->size()
            << " bytes, valid for " << infoH0F1.getStartValidityTimestamp()
            << " : " << infoH0F1.getEndValidityTimestamp();
  LOG(INFO) << "Sending object " << infoH1F0.getPath() << "/" << infoH1F0.getFileName()
            << " of size " << imageH1F0->size()
            << " bytes, valid for " << infoH1F0.getStartValidityTimestamp()
            << " : " << infoH1F0.getEndValidityTimestamp();
  LOG(INFO) << "Sending object " << infoH1F1.getPath() << "/" << infoH1F1.getFileName()
            << " of size " << imageH1F1->size()
            << " bytes, valid for " << infoH1F1.getStartValidityTimestamp()
            << " : " << infoH1F1.getEndValidityTimestamp();

  using clbUtils = o2::calibration::Utils;
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 0}, *imageH0F0.get());
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 0}, infoH0F0);
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 1}, *imageH0F1.get());
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 1}, infoH0F1);
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 2}, *imageH1F0.get());
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 2}, infoH1F0);
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 3}, *imageH1F1.get());
  output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 3}, infoH1F1);
}

void NoiseCalibratorDigitsSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  sendOutput(ec.outputs());
}

DataProcessorSpec getNoiseCalibratorDigitsSpec()
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginMFT;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);

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
    AlgorithmSpec{adaptFromTask<NoiseCalibratorDigitsSpec>()},
    Options{
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}},
      {"tstart", VariantType::Int64, -1ll, {"Start of validity timestamp"}},
      {"tend", VariantType::Int64, -1ll, {"End of validity timestamp"}},
//      {"path", VariantType::String, "/MFT/Calib/NoiseMap", {"Path to write to in CCDB"}},
      {"meta", VariantType::String, "", {"meta data to write in CCDB"}}}};
}

} // namespace mft
} // namespace o2
