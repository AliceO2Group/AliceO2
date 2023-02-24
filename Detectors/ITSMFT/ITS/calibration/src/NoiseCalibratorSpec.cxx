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
#include "CCDB/CCDBTimeStampUtils.h"
#include "DetectorsCalibration/Utils.h"
#include "ITSCalibration/NoiseCalibratorSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

void NoiseCalibratorSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  auto onepix = ic.options().get<bool>("1pix-only");
  LOG(info) << "Fast 1=pixel calibration: " << onepix;
  auto probT = ic.options().get<float>("prob-threshold");
  auto probTRelErr = ic.options().get<float>("prob-rel-err");
  mNoiseCutIB = ic.options().get<float>("cut-ib");
  LOGP(info, "Setting the probability threshold to {} with relative error {}", probT, probTRelErr);
  mStopMeOnly = ic.options().get<bool>("stop-me-only");
  mCalibrator = std::make_unique<CALIBRATOR>(onepix, probT, probTRelErr);
  mCalibrator->setNThreads(ic.options().get<int>("nthreads"));
  mValidityDays = ic.options().get<int>("validity-days");
  if (mValidityDays < 1) {
    mValidityDays = 1;
  }
}

void NoiseCalibratorSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  mTimer.Start(false);
  static bool firstCall = true;
  static bool done = false;
  if (done) {
    return;
  }
  if (firstCall) {
    firstCall = false;
    mCalibrator->setInstanceID((int)pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId);
    mCalibrator->setNInstances((int)pc.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices);
    if (mMode == ProcessingMode::Accumulate) {
      mCalibrator->setMinROFs(mCalibrator->getMinROFs() / mCalibrator->getNInstances());
    }
  }

  if (mMode == ProcessingMode::Full || mMode == ProcessingMode::Accumulate) {
    if (mUseClusters) {
      const auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
      gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
      const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
      mNClustersProc += compClusters.size();
      mDataSizeStat +=
        rofs.size() * sizeof(o2::itsmft::ROFRecord) + patterns.size() +
        compClusters.size() * sizeof(o2::itsmft::CompClusterExt);
      done = mCalibrator->processTimeFrameClusters(compClusters, patterns, rofs);
    } else {
      const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
      const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
      mNClustersProc += digits.size();
      mDataSizeStat += digits.size() * sizeof(o2::itsmft::Digit) + rofs.size() * sizeof(o2::itsmft::ROFRecord);
      done = mCalibrator->processTimeFrameDigits(digits, rofs);
    }
  } else {
    const auto extMap = pc.inputs().get<o2::itsmft::NoiseMap*>("mapspart");
    gsl::span<const int> partInfo = pc.inputs().get<gsl::span<int>>("mapspartInfo");
    mCalibrator->addMap(*extMap.get());
    done = (++mNPartsDone == partInfo[1]);
    mStrobeCounter += partInfo[2];
    mCalibrator->setNStrobes(mStrobeCounter);
    LOGP(info, "Received accumulated map {} of {} with {} ROFs, total number of maps = {} and strobes = {}", partInfo[0] + 1, partInfo[1], partInfo[2], mNPartsDone, mCalibrator->getNStrobes());
  }
  if (done) {
    LOG(info) << "Minimum number of noise counts has been reached !";
    if (mMode == ProcessingMode::Full || mMode == ProcessingMode::Normalize) {
      sendOutput(pc.outputs());
      // pc.services().get<ControlService>().readyToQuit(mStopMeOnly ? QuitRequest::Me : QuitRequest::All);
    } else {
      sendAccumulatedMap(pc.outputs());
      // pc.services().get<o2::framework::ControlService>().endOfStream();
    }
  }

  mTimer.Stop();
}

void NoiseCalibratorSpec::sendAccumulatedMap(DataAllocator& output)
{
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  output.snapshot(Output{"ITS", "NOISEMAPPART", (unsigned int)mCalibrator->getInstanceID()}, mCalibrator->getNoiseMap());
  std::vector<int> outInf;
  outInf.push_back(mCalibrator->getInstanceID());
  outInf.push_back(mCalibrator->getNInstances());
  outInf.push_back(mCalibrator->getNStrobes());
  output.snapshot(Output{"ITS", "NOISEMAPPARTINF", (unsigned int)mCalibrator->getInstanceID()}, outInf);
  LOGP(info, "Sending accumulated map with {} ROFs processed", mCalibrator->getNStrobes());
}

void NoiseCalibratorSpec::sendOutput(DataAllocator& output)
{
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  mCalibrator->finalize(mNoiseCutIB);

  long tstart = o2::ccdb::getCurrentTimestamp();
  long tend = o2::ccdb::getFutureTimestamp(3600 * 24 * mValidityDays);
#ifdef TIME_SLOT_CALIBRATION
  const auto& payload = mCalibrator->getNoiseMap(tstart, tend);
#else
  const auto& payload = mCalibrator->getNoiseMap();
#endif

  // Preparing the object for production CCDB and sending it
  std::map<std::string, std::string> md;

  o2::ccdb::CcdbObjectInfo info("ITS/Calib/NoiseMap", "NoiseMap", "noise.root", md, tstart, tend);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE", 0}, info);
  LOG(info) << "sending of o2::itsmft::NoiseMap done";

  // Preparing the object for DCS CCDB and sending it to output
  for (int ichip = 0; ichip < o2::itsmft::ChipMappingITS::getNChips(); ichip++) {
    const std::map<int, int>* chipmap = payload.getChipMap(ichip);
    if (chipmap) {
      for (std::map<int, int>::const_iterator it = chipmap->begin(); it != chipmap->end(); ++it) {
        addDatabaseEntry(ichip, payload.key2Row(it->first), payload.key2Col(it->first));
      }
    }
  }

  o2::ccdb::CcdbObjectInfo info_dcs("ITS/Calib/NOISE", "NoiseMap", "noise_scan.root", md, tstart, tend); // to DCS CCDB
  auto image_dcs = o2::ccdb::CcdbApi::createObjectImage(&mNoiseMapDCS, &info_dcs);
  info_dcs.setFileName("noise_scan.root");
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE", 1}, *image_dcs.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE", 1}, info_dcs);
  LOG(info) << "sending of DCSConfigObject done";

  // Timer
  LOGP(info, "Timing: {:.2f} CPU / {:.2f} Real s. in {} TFs for {} {} / {:.2f} GB",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1,
       mUseClusters ? "clusters" : "digits",
       mNClustersProc, double(mDataSizeStat) / (1024L * 1024L * 1024L));
}

void NoiseCalibratorSpec::addDatabaseEntry(int chip, int row, int col)
{
  o2::dcs::addConfigItem(mNoiseMapDCS, "O2ChipID", std::to_string(chip));
  o2::dcs::addConfigItem(mNoiseMapDCS, "ChipDbID", std::to_string(mConfDBmap->at(chip)));
  o2::dcs::addConfigItem(mNoiseMapDCS, "Dcol", "-1"); // dummy, just to keep the same format between digital scan and noise scan (easier dcs scripts)
  o2::dcs::addConfigItem(mNoiseMapDCS, "Row", std::to_string(row));
  o2::dcs::addConfigItem(mNoiseMapDCS, "Col", std::to_string(col));
}

void NoiseCalibratorSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mMode == ProcessingMode::Accumulate) {
    sendAccumulatedMap(ec.outputs());
  } else {
    sendOutput(ec.outputs());
  }
}

///_______________________________________
void NoiseCalibratorSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    if (mUseClusters) {
      pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    }
    if (mMode == ProcessingMode::Full || mMode == ProcessingMode::Normalize) {
      pc.inputs().get<std::vector<int>*>("confdbmap");
    }
  }
}

///_______________________________________
void NoiseCalibratorSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mCalibrator->setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }

  if (matcher == ConcreteDataMatcher("ITS", "CONFDBMAP", 0)) {
    LOG(info) << "confDB map updated";
    mConfDBmap = (std::vector<int>*)obj;
    return;
  }
}

DataProcessorSpec getNoiseCalibratorSpec(bool useClusters, int pmode)
{
  NoiseCalibratorSpec::ProcessingMode md = NoiseCalibratorSpec::ProcessingMode::Full;
  std::string name = "its-noise-calibrator";
  if (pmode == int(NoiseCalibratorSpec::ProcessingMode::Full)) {
    md = NoiseCalibratorSpec::ProcessingMode::Full;
  } else if (pmode == int(NoiseCalibratorSpec::ProcessingMode::Accumulate)) {
    md = NoiseCalibratorSpec::ProcessingMode::Accumulate;
  } else if (pmode == int(NoiseCalibratorSpec::ProcessingMode::Normalize)) {
    md = NoiseCalibratorSpec::ProcessingMode::Normalize;
    name = "its-noise-calibrator_Norm";
  } else {
    LOG(fatal) << "Unknown processing mode " << pmode;
  }

  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  if (md == NoiseCalibratorSpec::ProcessingMode::Full || md == NoiseCalibratorSpec::ProcessingMode::Accumulate) {
    if (useClusters) {
      inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
      inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
      inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
      inputs.emplace_back("cldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
    } else {
      inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
      inputs.emplace_back("ROframes", "ITS", "DIGITSROF", 0, Lifetime::Timeframe);
    }
  } else {
    useClusters = false;                                                                                         // not needed for normalization
    inputs.emplace_back("mapspart", ConcreteDataTypeMatcher{"ITS", "NOISEMAPPART"}, Lifetime::Timeframe);        // for normalization of multiple inputs only
    inputs.emplace_back("mapspartInfo", ConcreteDataTypeMatcher{"ITS", "NOISEMAPPARTINF"}, Lifetime::Timeframe); // for normalization of multiple inputs only
  }
  if (md == NoiseCalibratorSpec::ProcessingMode::Full || md == NoiseCalibratorSpec::ProcessingMode::Normalize) {
    inputs.emplace_back("confdbmap", "ITS", "CONFDBMAP", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/Confdbmap"));

    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE"}, Lifetime::Sporadic);
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE"}, Lifetime::Sporadic);
  } else { // in accumulation mode the output is a map
    outputs.emplace_back(ConcreteDataTypeMatcher{"ITS", "NOISEMAPPART"}, Lifetime::Sporadic);
    outputs.emplace_back(ConcreteDataTypeMatcher{"ITS", "NOISEMAPPARTINF"}, Lifetime::Sporadic);
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                false,                          // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  return DataProcessorSpec{
    name,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(md, useClusters, ccdbRequest)},
    Options{
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only (cluster input only)"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}},
      {"prob-rel-err", VariantType::Float, 0.2f, {"Relative error on channel noise to apply the threshold"}},
      {"cut-ib", VariantType::Float, -1.f, {"Special cut to apply to Inner Barrel"}},
      {"nthreads", VariantType::Int, 1, {"Number of map-filling threads"}},
      {"validity-days", VariantType::Int, 3, {"Validity on days from upload time"}},
      {"stop-me-only", VariantType::Bool, false, {"At sufficient statistics stop only this device, otherwise whole workflow"}}}};
}

} // namespace its
} // namespace o2
