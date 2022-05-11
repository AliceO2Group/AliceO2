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

/// @file   NoiseCalibratorSpec.cxx

#include "CCDB/CcdbApi.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCalibration/Utils.h"
#include "MFTCalibration/NoiseCalibratorSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSMFTReconstruction/ClustererParam.h"

#include "FairLogger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

using namespace o2::framework;
using namespace o2::utils;

namespace o2
{
namespace mft
{

NoiseCalibratorSpec::NoiseCalibratorSpec(bool useDigits, std::shared_ptr<o2::base::GRPGeomRequest> req)
  : mDigits(useDigits), mCCDBRequest(req)
{
}

void NoiseCalibratorSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  auto probT = ic.options().get<float>("prob-threshold");
  auto probTRelErr = ic.options().get<float>("prob-rel-err");
  LOGP(info, "Setting the probability threshold to {} with relative error {}", probT, probTRelErr);
  mStopMeOnly = ic.options().get<bool>("stop-me-only");
  mPath = ic.options().get<std::string>("path-CCDB");
  mMeta = ic.options().get<std::string>("meta");
  mStart = ic.options().get<int64_t>("tstart");
  mEnd = ic.options().get<int64_t>("tend");

  mCalibrator = std::make_unique<CALIBRATOR>(probT, probTRelErr);

  mPathDcs = ic.options().get<std::string>("path-DCS");
  mOutputType = ic.options().get<std::string>("send-to-server");
  mNoiseMapForDcs.clear();
}

void NoiseCalibratorSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  if (mDigits) {
    const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digitsROF").header)->startTime;

    if (mCalibrator->processTimeFrame(tfcounter, digits, rofs)) {
      LOG(info) << "Minimum number of noise counts has been reached !";
      if (mOutputType.compare("CCDB") == 0) {
        LOG(info) << "Sending an object to Production-CCDB";
        sendOutputCcdb(pc.outputs());
      } else if (mOutputType.compare("DCS") == 0) {
        LOG(info) << "Sending an object to DCS-CCDB";
        sendOutputDcs(pc.outputs());
      } else {
        LOG(info) << "Sending an object to Production-CCDB and DCS-CCDB";
        sendOutputCcdbDcs(pc.outputs());
      }
      pc.services().get<ControlService>().readyToQuit(mStopMeOnly ? QuitRequest::Me : QuitRequest::All);
    }
  } else {
    const auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
    gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("ROframes").header)->startTime;

    if (mCalibrator->processTimeFrame(tfcounter, compClusters, patterns, rofs)) {
      LOG(info) << "Minimum number of noise counts has been reached !";
      if (mOutputType.compare("CCDB") == 0) {
        LOG(info) << "Sending an object to Production-CCDB";
        sendOutputCcdb(pc.outputs());
      } else if (mOutputType.compare("DCS") == 0) {
        LOG(info) << "Sending an object to DCS-CCDB";
        sendOutputDcs(pc.outputs());
      } else {
        LOG(info) << "Sending an object to Production-CCDB and DCS-CCDB";
        sendOutputCcdbDcs(pc.outputs());
      }
      pc.services().get<ControlService>().readyToQuit(mStopMeOnly ? QuitRequest::Me : QuitRequest::All);
    }
  }
}

void NoiseCalibratorSpec::setOutputDcs(const o2::itsmft::NoiseMap& payload)
{
  for (int iChip = 0; iChip < 936; ++iChip) {
    for (int iRow = 0; iRow < 512; ++iRow) {
      for (int iCol = 0; iCol < 1024; ++iCol) {

        if (!payload.isNoisy(iChip, iRow, iCol)) {
          continue;
        }
        std::array<int, 4> noise = {iChip, iRow, iCol, 0};
        mNoiseMapForDcs.emplace_back(noise);
      }
    }
  }
}

void NoiseCalibratorSpec::sendOutputCcdbDcs(DataAllocator& output)
{

  LOG(info) << "CCDB-DCS mode";

  static bool done = false;
  if (done) {
    return;
  }
  done = true;

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
        LOG(error) << "Illegal command-line key/value string: " << token;
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
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  using clbUtils = o2::calibration::Utils;
  output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MFT_NoiseMap", 0}, *image.get());
  output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MFT_NoiseMap", 0}, info);

  setOutputDcs(payload);

  o2::ccdb::CcdbObjectInfo infoDcs(mPathDcs, "NoiseMap", "noise.root", meta, tstart, tend);
  auto flNameDcs = o2::ccdb::CcdbApi::generateFileName("noise");
  auto imageDcs = o2::ccdb::CcdbApi::createObjectImage(&mNoiseMapForDcs, &infoDcs);
  infoDcs.setFileName(flNameDcs);
  LOG(info) << "Sending object " << infoDcs.getPath() << "/" << infoDcs.getFileName()
            << " of size " << imageDcs->size()
            << " bytes, valid for " << infoDcs.getStartValidityTimestamp()
            << " : " << infoDcs.getEndValidityTimestamp();

  using clbUtilsDcs = o2::calibration::Utils;
  output.snapshot(Output{clbUtilsDcs::gDataOriginCDBPayload, "MFT_NoiseMap", 1}, *imageDcs.get());
  output.snapshot(Output{clbUtilsDcs::gDataOriginCDBWrapper, "MFT_NoiseMap", 1}, infoDcs);
}

void NoiseCalibratorSpec::sendOutputCcdb(DataAllocator& output)
{

  LOG(info) << "CCDB mode";

  static bool done = false;
  if (done) {
    return;
  }
  done = true;

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
        LOG(error) << "Illegal command-line key/value string: " << token;
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
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  using clbUtils = o2::calibration::Utils;
  output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MFT_NoiseMap", 0}, *image.get());
  output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MFT_NoiseMap", 0}, info);
}

void NoiseCalibratorSpec::sendOutputDcs(DataAllocator& output)
{

  LOG(info) << "DCS mode";

  static bool done = false;
  if (done) {
    return;
  }
  done = true;

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
        LOG(error) << "Illegal command-line key/value string: " << token;
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

  setOutputDcs(payload);

  o2::ccdb::CcdbObjectInfo infoDcs(mPathDcs, "NoiseMap", "noise.root", meta, tstart, tend);
  auto flNameDcs = o2::ccdb::CcdbApi::generateFileName("noise");
  auto imageDcs = o2::ccdb::CcdbApi::createObjectImage(&mNoiseMapForDcs, &infoDcs);
  infoDcs.setFileName(flNameDcs);
  LOG(info) << "Sending object " << infoDcs.getPath() << "/" << infoDcs.getFileName()
            << " of size " << imageDcs->size()
            << " bytes, valid for " << infoDcs.getStartValidityTimestamp()
            << " : " << infoDcs.getEndValidityTimestamp();

  using clbUtilsDcs = o2::calibration::Utils;
  output.snapshot(Output{clbUtilsDcs::gDataOriginCDBPayload, "MFT_NoiseMap", 0}, *imageDcs.get());
  output.snapshot(Output{clbUtilsDcs::gDataOriginCDBWrapper, "MFT_NoiseMap", 0}, infoDcs);
}

void NoiseCalibratorSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mOutputType.compare("CCDB") == 0) {
    LOG(info) << "Sending an object to Production-CCDB";
    sendOutputCcdb(ec.outputs());
  } else if (mOutputType.compare("DCS") == 0) {
    LOG(info) << "Sending an object to DCS-CCDB";
    sendOutputDcs(ec.outputs());
  } else {
    LOG(info) << "Sending an object to Production-CCDB and DCS-CCDB";
    sendOutputCcdbDcs(ec.outputs());
  }
}

///_______________________________________
void NoiseCalibratorSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!mDigits) {
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
  }
}

///_______________________________________
void NoiseCalibratorSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mCalibrator->setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
  }
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
    inputs.emplace_back("cldict", "MFT", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/ClusterDictionary"));
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                false,                          // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "MFT_NoiseMap"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "MFT_NoiseMap"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "mft-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(useDigits, ccdbRequest)},
    Options{
      {"prob-threshold", VariantType::Float, 1.e-6f, {"Probability threshold for noisy pixels"}},
      {"prob-rel-err", VariantType::Float, 0.2f, {"Relative error on channel noise to apply the threshold"}},
      {"tstart", VariantType::Int64, -1ll, {"Start of validity timestamp"}},
      {"tend", VariantType::Int64, -1ll, {"End of validity timestamp"}},
      {"path-CCDB", VariantType::String, "/MFT/Calib/NoiseMap", {"Path to write to in CCDB"}},
      {"path-DCS", VariantType::String, "/MFT/Config/NoiseMap", {"Path to write to in CCDB"}},
      {"meta", VariantType::String, "", {"meta data to write in CCDB"}},
      {"send-to-server", VariantType::String, "CCDB-DCS", {"meta data to write in DCS-CCDB"}},
      {"stop-me-only", VariantType::Bool, false, {"At sufficient statistics stop only this device, otherwise whole workflow"}}}};
}

} // namespace mft
} // namespace o2
