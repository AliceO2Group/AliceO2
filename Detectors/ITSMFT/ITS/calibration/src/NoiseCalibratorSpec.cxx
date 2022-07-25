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
  if (firstCall) {
    firstCall = false;
    mCalibrator->setInstanceID(pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId);
    mCalibrator->setNInstances(pc.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices);
  }
  bool done = false;
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
  if (done) {
    LOG(info) << "Minimum number of noise counts has been reached !";
    sendOutput(pc.outputs());
    pc.services().get<ControlService>().readyToQuit(mStopMeOnly ? QuitRequest::Me : QuitRequest::All);
  }

  mTimer.Stop();
}

void NoiseCalibratorSpec::sendOutput(DataAllocator& output)
{
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  mCalibrator->finalize();

  long tstart = o2::ccdb::getCurrentTimestamp();
  long tend = o2::ccdb::getFutureTimestamp(3600 * 24 * mValidityDays);
#ifdef TIME_SLOT_CALIBRATION
  const auto& payload = mCalibrator->getNoiseMap(tstart, tend);
#else
  const auto& payload = mCalibrator->getNoiseMap();
#endif
  std::map<std::string, std::string> md;
  o2::ccdb::CcdbObjectInfo info("ITS/Calib/NoiseMap", "NoiseMap", "noise.root", md, tstart, tend);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE", 0}, info);
  LOG(info) << "sending done";
  LOGP(info, "Timing: {:.2f} CPU / {:.2f} Real s. in {} TFs for {} {} / {:.2f} GB",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1,
       mUseClusters ? "clusters" : "digits",
       mNClustersProc, double(mDataSizeStat) / (1024L * 1024L * 1024L));
}

void NoiseCalibratorSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  sendOutput(ec.outputs());
}

///_______________________________________
void NoiseCalibratorSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (mUseClusters) {
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
  }
}

///_______________________________________
void NoiseCalibratorSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mCalibrator->setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
  }
}

DataProcessorSpec getNoiseCalibratorSpec(bool useClusters)
{
  std::vector<InputSpec> inputs;
  if (useClusters) {
    inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
    inputs.emplace_back("cldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  } else {
    inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "ITS", "DIGITSROF", 0, Lifetime::Timeframe);
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                false,                          // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "its-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(useClusters, ccdbRequest)},
    Options{
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only (cluster input only)"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}},
      {"prob-rel-err", VariantType::Float, 0.2f, {"Relative error on channel noise to apply the threshold"}},
      {"nthreads", VariantType::Int, 1, {"Number of map-filling threads"}},
      {"validity-days", VariantType::Int, 3, {"Validity on days from upload time"}},
      {"stop-me-only", VariantType::Bool, false, {"At sufficient statistics stop only this device, otherwise whole workflow"}}}};
}

} // namespace its
} // namespace o2
