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
#include "CommonUtils/StringUtils.h"
#include "DetectorsCalibration/Utils.h"
#include "MFTCalibration/NoiseCalibratorSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "FairLogger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;
using namespace o2::utils;

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
  auto probT = ic.options().get<float>("prob-threshold");
  LOG(INFO) << "Setting the probability threshold to " << probT;

  mPath = ic.options().get<std::string>("path");
  mMeta = ic.options().get<std::string>("meta");
  mStart = ic.options().get<int64_t>("tstart");
  mEnd = ic.options().get<int64_t>("tend");

  mCalibrator = std::make_unique<CALIBRATOR>(probT);
}

void NoiseCalibratorSpec::run(ProcessingContext& pc)
{
  if (mDigits) {
    const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digitsROF").header)->startTime;

    if (mCalibrator->processTimeFrame(tfcounter, digits, rofs)) {
      LOG(INFO) << "Minimum number of noise counts has been reached !";
      sendOutput(pc.outputs());
      pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
    }
  } else {
    const auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
    gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("ROframes").header)->startTime;

    if (mCalibrator->processTimeFrame(tfcounter, compClusters, patterns, rofs)) {
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

  std::map<std::string, std::string> meta;
  auto toKeyValPairs = [&meta](std::vector<std::string> const& tokens) {
    for (auto& token : tokens) {
      auto keyval = Str::tokenize(token, '=', false);
      if (keyval.size() != 2) {
        LOG(ERROR) << "Illegal command-line key/value string: " << token;
        continue;
      }
      Str::trim(keyval[1]);
      meta[keyval[0]] = keyval[1];
    }
  };
  toKeyValPairs(Str::tokenize(mMeta, ';', true));

  long startTF, endTF;

  const auto& payload = mCalibrator->getNoiseMap();
  //  const auto& payload = mCalibrator->getNoiseMap(starTF, endTF); //For TimeSlot calibration

  o2::ccdb::CcdbObjectInfo info(mPath, "NoiseMap", "noise.root", meta, tstart, tend);
  auto flName = o2::ccdb::CcdbApi::generateFileName("noise");
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  info.setFileName(flName);
  LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  using clbUtils = o2::calibration::Utils;
  output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MFT_NoiseMap", 0}, *image.get());
  output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MFT_NoiseMap", 0}, info);
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
  } else {
    inputs.emplace_back("compClusters", detOrig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", detOrig, "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", detOrig, "CLUSTERSROF", 0, Lifetime::Timeframe);
  }

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "MFT_NoiseMap"});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "MFT_NoiseMap"});

  return DataProcessorSpec{
    "mft-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(useDigits)},
    Options{
      {"prob-threshold", VariantType::Float, 1.e-6f, {"Probability threshold for noisy pixels"}},
      {"tstart", VariantType::Int64, -1ll, {"Start of validity timestamp"}},
      {"tend", VariantType::Int64, -1ll, {"End of validity timestamp"}},
      {"path", VariantType::String, "/MFT/Calib/NoiseMap", {"Path to write to in CCDB"}},
      {"meta", VariantType::String, "", {"meta data to write in CCDB"}},
      {"hb-per-tf", VariantType::Int, 256, {"Number of HBF per TF"}}}};
}

} // namespace mft
} // namespace o2
