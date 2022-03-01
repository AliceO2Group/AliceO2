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
#include "DetectorsCalibration/Utils.h"
#include "ITSCalibration/NoiseCalibratorSpec.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
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
  auto onepix = ic.options().get<bool>("1pix-only");
  LOG(info) << "Fast 1=pixel calibration: " << onepix;
  auto probT = ic.options().get<float>("prob-threshold");
  LOG(info) << "Setting the probability threshold to " << probT;

  mCalibrator = std::make_unique<CALIBRATOR>(onepix, probT);
  mCalibrator->setNThreads(ic.options().get<int>("nthreads"));

  std::string dictPath = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().dictFilePath;
  std::string dictFile = o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, dictPath);
  if (o2::utils::Str::pathExists(dictFile)) {
    mCalibrator->loadDictionary(dictFile);
    LOG(info) << "ITS NoiseCalibrator is running with a provided dictionary: " << dictFile;
  } else {
    LOG(info) << "Dictionary " << dictFile
              << " is absent, ITS NoiseCalibrator expects cluster patterns for all clusters";
  }
}

void NoiseCalibratorSpec::run(ProcessingContext& pc)
{
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
    pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
  }

  mTimer.Stop();
}

void NoiseCalibratorSpec::sendOutput(DataAllocator& output)
{
  mCalibrator->finalize();

  long tstart = 0, tend = 9999999;
#ifdef TIME_SLOT_CALIBRATION
  const auto& payload = mCalibrator->getNoiseMap(tstart, tend);
#else
  const auto& payload = mCalibrator->getNoiseMap();
#endif
  std::map<std::string, std::string> md;
  o2::ccdb::CcdbObjectInfo info("ITS/Noise", "NoiseMap", "noise.root", md, tstart, tend);

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

DataProcessorSpec getNoiseCalibratorSpec(bool useClusters)
{
  std::vector<InputSpec> inputs;
  if (useClusters) {
    inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  } else {
    inputs.emplace_back("digits", "ITS", "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "ITS", "DIGITSROF", 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_NOISE"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_NOISE"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "its-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibratorSpec>(useClusters)},
    Options{
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only (cluster input only)"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}},
      {"nthreads", VariantType::Int, 1, {"Number of map-filling threads"}}}};
}

} // namespace its
} // namespace o2
