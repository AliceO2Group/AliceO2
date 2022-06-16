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

/// @file   DigitRecoSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ZDCWorkflow/DigitRecoSpec.h"
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
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"
#include "ZDCReconstruction/BaselineParam.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

DigitRecoSpec::DigitRecoSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

DigitRecoSpec::DigitRecoSpec(const int verbosity, const bool debugOut)
  : mVerbosity(verbosity), mDebugOut(debugOut)
{
  mTimer.Stop();
  mTimer.Reset();
}

void DigitRecoSpec::init(o2::framework::InitContext& ic)
{
  mEnableBaselineParam = ic.options().get<bool>("disable-baseline-calib");
  mEnableZDCTDCCorr = ic.options().get<bool>("disable-tdc-corr");
  mEnableZDCEnergyParam = ic.options().get<bool>("disable-energy-calib");
  mEnableZDCTowerParam = ic.options().get<bool>("disable-tower-calib");
  mccdbHost = ic.options().get<std::string>("ccdb-url");
}

void DigitRecoSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
  pc.inputs().get<o2::zdc::RecoConfigZDC*>("recoconfig");
  pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
  if (mEnableZDCTDCCorr) {
    pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccorr");
  }
  if (mEnableZDCEnergyParam) {
    pc.inputs().get<o2::zdc::ZDCTDCParam*>("adccalib");
  }

  LOG(info) << "mEnableBaselineParam=" <<  mEnableBaselineParam;
  if (mEnableBaselineParam) {
    pc.inputs().get<o2::zdc::BaselineParam*>("basecalib");
  }
}

void DigitRecoSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  if (!mInitialized) {
    mInitialized = true;
    std::string loadedConfFiles = "Loaded ZDC configuration files:";
    {
      // Module configuration
      auto config = pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
      loadedConfFiles += " Moduleconfig";
      mWorker.setModuleConfig(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    }
    {
      // Configuration parameters for ZDC reconstruction
      auto config = pc.inputs().get<o2::zdc::RecoConfigZDC*>("recoconfig");
      loadedConfFiles += " RecoConfigZDC";
      mWorker.setRecoConfigZDC(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    }
    {
      // TDC centering
      auto config = pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
      loadedConfFiles += " ZDCTDCParam";
      mWorker.setTDCParam(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    }
    if (mEnableZDCTDCCorr) {
      // TDC correction parameters
      auto config = pc.inputs().get<o2::zdc::ZDCTDCCorr*>("tdccorr");
      loadedConfFiles += " ZDCTDCCorr";
      mWorker.setTDCCorr(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    } else {
      LOG(warning) << "ZDCTDCCorr has been disabled - no correction is applied";
    }
    if (mEnableZDCEnergyParam) {
      // Energy calibration
      auto config = pc.inputs().get<o2::zdc::ZDCEnergyParam*>("adccalib");
      loadedConfFiles += " ZDCEnergyParam";
      mWorker.setEnergyParam(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    } else {
      LOG(warning) << "ZDCEnergyParam has been disabled - no energy calibration is applied";
    }
    if (mEnableZDCTowerParam) {
      // Tower intercalibration
      auto config = pc.inputs().get<o2::zdc::ZDCTowerParam*>("towercalib");
      loadedConfFiles += " ZDCTowerParam";
      mWorker.setTowerParam(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    } else {
      LOG(warning) << "ZDCTowerParam has been disabled - no tower intercalibration";
    }
    if (mEnableBaselineParam) {
      // Average pedestals
      auto config = pc.inputs().get<o2::zdc::BaselineParam*>("basecalib");
      loadedConfFiles += " BaselineParam";
      mWorker.setBaselineParam(config.get());
      if (mVerbosity > DbgZero) {
        config->print();
      }
    } else {
      LOG(warning) << "BaselineParam has been disabled - no fallback in case orbit pedestals are missing";
    }
    LOG(info) << loadedConfFiles;
    if (mDebugOut) {
      mWorker.setDebugOutput();
    }
    mWorker.setVerbosity(mVerbosity);
    mWorker.init();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcdata = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chans = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  mWorker.process(peds, bcdata, chans);
  const std::vector<o2::zdc::RecEventAux>& recAux = mWorker.getReco();

  // Transfer wafeform
  bool fullinter = mWorker.getFullInterpolation();
  RecEvent recEvent;
  LOGF(info, "BC processed during reconstruction %d%s", recAux.size(), (fullinter ? " FullInterpolation" : ""));
  uint32_t nte = 0, ntt = 0, nti = 0, ntw = 0;
  for (auto reca : recAux) {
    bool toAddBC = true;
    int32_t ne = reca.ezdc.size();
    int32_t nt = 0;
    // Store TDC hits
    for (int32_t it = 0; it < o2::zdc::NTDCChannels; it++) {
      for (int32_t ih = 0; ih < reca.ntdc[it]; ih++) {
        if (toAddBC) {
          recEvent.addBC(reca);
          toAddBC = false;
        }
        nt++;
        recEvent.addTDC(it, reca.TDCVal[it][ih], reca.TDCAmp[it][ih], reca.isBeg[it], reca.isEnd[it]);
      }
    }
    // Add waveform information
    if (fullinter) {
      for (int32_t isig = 0; isig < o2::zdc::NChannels; isig++) {
        if (reca.inter[isig].size() == NIS) {
          if (toAddBC) {
            recEvent.addBC(reca);
            toAddBC = false;
          }
          recEvent.addWaveform(isig, reca.inter[isig]);
          ntw++;
        }
      }
    }
    if (ne > 0) {
      if (toAddBC) {
        recEvent.addBC(reca);
        toAddBC = false;
      }
      std::map<uint8_t, float>::iterator it;
      for (it = reca.ezdc.begin(); it != reca.ezdc.end(); it++) {
        recEvent.addEnergy(it->first, it->second);
      }
    }
    nte += ne;
    ntt += nt;
    if (mVerbosity > 1 && (nt > 0 || ne > 0)) {
      printf("Orbit %9u bc %4u ntdc %2d ne %2d channels=0x%08x\n", reca.ir.orbit, reca.ir.bc, nt, ne, reca.channels);
    }
    // Event information
    nti += recEvent.addInfos(reca);
  }
  LOG(info) << "Reconstructed " << ntt << " signal TDCs and " << nte << " ZDC energies and "
            << nti << " info messages in " << recEvent.mRecBC.size() << "/" << recAux.size() << " b.c. and "
            << ntw << " waveform chunks";
  // TODO: rate information for all channels
  // TODO: summary of reconstruction to be collected by DQM?
  pc.outputs().snapshot(Output{"ZDC", "BCREC", 0, Lifetime::Timeframe}, recEvent.mRecBC);
  pc.outputs().snapshot(Output{"ZDC", "ENERGY", 0, Lifetime::Timeframe}, recEvent.mEnergy);
  pc.outputs().snapshot(Output{"ZDC", "TDCDATA", 0, Lifetime::Timeframe}, recEvent.mTDCData);
  pc.outputs().snapshot(Output{"ZDC", "INFO", 0, Lifetime::Timeframe}, recEvent.mInfo);
  pc.outputs().snapshot(Output{"ZDC", "WAVE", 0, Lifetime::Timeframe}, recEvent.mWaveform);
  mTimer.Stop();
}

void DigitRecoSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.eor();
  LOGF(info, "ZDC Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getDigitRecoSpec(const int verbosity = 0, const bool enableDebugOut = false)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathConfigModule.data())));
  inputs.emplace_back("recoconfig", "ZDC", "RECOCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathRecoConfigZDC.data())));
  inputs.emplace_back("tdccalib", "ZDC", "TDCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalib.data())));
  inputs.emplace_back("tdccorr", "ZDC", "TDCCORR", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCorr.data())));
  inputs.emplace_back("adccalib", "ZDC", "ADCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathEnergyCalib.data())));
  inputs.emplace_back("towercalib", "ZDC", "TOWERCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTowerCalib.data())));
  inputs.emplace_back("basecalib", "ZDC", "BASECALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathBaselineCalib.data())));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "BCREC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "ENERGY", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "INFO", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "WAVE", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "zdc-digi-reco",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DigitRecoSpec>(verbosity, enableDebugOut)},
    o2::framework::Options{
      {"ccdb-url", o2::framework::VariantType::String, o2::base::NameConf::getCCDBServer(), {"CCDB Url"}},
      {"disable-tdc-corr", o2::framework::VariantType::Bool, false, {"Get ZDCTDCCorr calibration object"}},
      {"disable-energy-calib", o2::framework::VariantType::Bool, false, {"Get ZDCEnergyParam calibration object"}},
      {"disable-tower-calib", o2::framework::VariantType::Bool, false, {"Get ZDCTowerParam calibration object"}},
      {"disable-baseline-calib", o2::framework::VariantType::Bool, false, {"Get BaselineParam calibration object"}}}};
}

} // namespace zdc
} // namespace o2
