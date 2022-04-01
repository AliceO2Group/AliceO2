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

/// @file   InterCalibSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

InterCalibSpec::InterCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

InterCalibSpec::InterCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void InterCalibSpec::init(o2::framework::InitContext& ic)
{
  //     int minEnt = std::max(300, ic.options().get<int>("min-entries"));
  //     int nb = std::max(500, ic.options().get<int>("nbins"));
  //     int slotL = ic.options().get<int>("tf-per-slot");
  //     int delay = ic.options().get<int>("max-delay");
  mVerbosity = ic.options().get<int>("verbosity-level");
  mInterCalib.setVerbosity(mVerbosity);
  mTimer.CpuTime();
  mTimer.Start(false);
}

void InterCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  std::string loadedConfFiles = "Loaded ZDC configuration files:";
  // Energy calibration
  auto energyParam = pc.inputs().get<o2::zdc::ZDCEnergyParam*>("energycalib");
  if (!energyParam) {
    LOG(fatal) << "Missing ZDCEnergyParam calibration object";
    return;
  } else {
    loadedConfFiles += " ZDCEnergyParam";
    if (mVerbosity > DbgMinimal) {
      LOG(info) << "Loaded Energy calibration ZDCEnergyParam";
      energyParam->print();
    }
  }

  // Tower calibration
  auto towerParam = pc.inputs().get<o2::zdc::ZDCTowerParam*>("towercalib");
  if (!towerParam) {
    LOG(fatal) << "Missing ZDCTowerParam calibration object";
    return;
  } else {
    loadedConfFiles += " ZDCTowerParam";
    if (mVerbosity > DbgMinimal) {
      LOG(info) << "Loaded Tower calibration ZDCTowerParam";
      towerParam->print();
    }
  }

  // InterCalib configuration
  auto interConfig = pc.inputs().get<o2::zdc::InterCalibConfig*>("intercalibconfig");
  if (!interConfig) {
    LOG(fatal) << "Missing InterCalibConfig calibration InterCalibConfig";
    return;
  } else {
    loadedConfFiles += " InterCalibConfig";
    if (mVerbosity > DbgMinimal) {
      LOG(info) << "Loaded InterCalib configuration object";
      interConfig->print();
    }
  }

  LOG(info) << loadedConfFiles;

  mInterCalib.setEnergyParam(energyParam.get());
  mInterCalib.setTowerParam(towerParam.get());
  mInterCalib.setInterCalibConfig(interConfig.get());
}

void InterCalibSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  auto data = pc.inputs().get<InterCalibData>("intercalibdata");
  mInterCalib.process(data);
  auto h1 = pc.inputs().get<std::array<o2::dataformats::FlatHisto1D<float>*, 2 * InterCalibData::NH>>("intercalib1dh");
  auto h2 = pc.inputs().get<std::array<o2::dataformats::FlatHisto2D<float>*, InterCalibData::NH>>("intercalib2dh");
  mInterCalib.add(h1);
}

void InterCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mInterCalib.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC Intercalibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

//________________________________________________________________
void InterCalibSpec::sendOutput(o2::framework::DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  //     const auto& payloadVec = mCalibrator->getLHCphaseVector();
  //     auto& infoVec = mCalibrator->getLHCphaseInfoVector(); // use non-const version as we update it
  //     assert(payloadVec.size() == infoVec.size());
  //
  //     for (uint32_t i = 0; i < payloadVec.size(); i++) {
  //       auto& w = infoVec[i];
  //       auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
  //       LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
  //                 << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
  //       output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LHCphase", i}, *image.get()); // vector<char>
  //       output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LHCphase", i}, w);            // root-serialized
  //     }
  //     if (payloadVec.size()) {
  //       mCalibrator->initOutput(); // reset the outputs once they are already sent
  //     }
}

framework::DataProcessorSpec getInterCalibSpec()
{
  using device = o2::zdc::InterCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("intercalibconfig", "ZDC", "INTERCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathInterCalibConfig.data())));
  inputs.emplace_back("energycalib", "ZDC", "ENERGYCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathEnergyCalib.data())));
  inputs.emplace_back("towercalib", "ZDC", "TOWERCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTowerCalib.data())));
  inputs.emplace_back("intercalibdata", "ZDC", "INTERCALIBDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("intercalib1dh", "ZDC", "INTERCALIB1DH", 0, Lifetime::Timeframe);
  inputs.emplace_back("intercalib2dh", "ZDC", "INTERCALIB2DH", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_Intercalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_Intercalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-towers",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
      {"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
