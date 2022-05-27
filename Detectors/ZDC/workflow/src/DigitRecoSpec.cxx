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
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
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
  mccdbHost = ic.options().get<std::string>("ccdb-url");
}

void DigitRecoSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    // Initialization from CCDB
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(mccdbHost);
    /*long timeStamp = 0; // TIMESTAMP SHOULD NOT BE 0
    if (timeStamp == mgr.getTimestamp()) {
      return;
    }
    mgr.setTimestamp(timeStamp);*/

    std::string loadedConfFiles = "Loaded ZDC configuration files for timestamp " + std::to_string(mgr.getTimestamp()) + ":";
    auto* moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(fatal) << "Missing ModuleConfig ZDC configuration object";
      return;
    } else {
      loadedConfFiles += " ModuleConfig";
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded ZDC module configuration for timestamp " << mgr.getTimestamp();
      moduleConfig->print();
    }

    // Configuration parameters for ZDC reconstruction
    auto* recoConfigZDC = mgr.get<o2::zdc::RecoConfigZDC>(o2::zdc::CCDBPathRecoConfigZDC);
    if (!recoConfigZDC) {
      LOG(info) << loadedConfFiles;
      LOG(fatal) << "Missing RecoConfigZDC object";
      return;
    } else {
      loadedConfFiles += " RecoConfigZDC";
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded RecoConfigZDC for timestamp " << mgr.getTimestamp();
      recoConfigZDC->print();
    }

    // TDC centering
    auto* tdcParam = mgr.get<o2::zdc::ZDCTDCParam>(o2::zdc::CCDBPathTDCCalib);
    if (!tdcParam) {
      LOG(info) << loadedConfFiles;
      LOG(fatal) << "Missing ZDCTDCParam calibration object";
      return;
    } else {
      loadedConfFiles += " ZDCTDCParam";
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded TDC centering ZDCTDCParam for timestamp " << mgr.getTimestamp();
      tdcParam->print();
    }

    // TDC correction parameters
    auto* tdcCorr = mgr.get<o2::zdc::ZDCTDCCorr>(o2::zdc::CCDBPathTDCCorr);
    if (!tdcCorr) {
      LOG(warning) << "Missing ZDCTDCCorr calibration object - no correction is applied";
    } else {
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded TDC correction parameters for timestamp " << mgr.getTimestamp();
        tdcCorr->print();
      }
      loadedConfFiles += " ZDCTDCCorr";
    }

    // Energy calibration
    auto* energyParam = mgr.get<o2::zdc::ZDCEnergyParam>(o2::zdc::CCDBPathEnergyCalib);
    if (!energyParam) {
      LOG(warning) << "Missing ZDCEnergyParam calibration object - using default";
    } else {
      loadedConfFiles += " ZDCEnergyParam";
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded Energy calibration ZDCEnergyParam for timestamp " << mgr.getTimestamp();
        energyParam->print();
      }
    }

    // Tower calibration
    auto* towerParam = mgr.get<o2::zdc::ZDCTowerParam>(o2::zdc::CCDBPathTowerCalib);
    if (!towerParam) {
      LOG(warning) << "Missing ZDCTowerParam calibration object - using default";
    } else {
      loadedConfFiles += " ZDCTowerParam";
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded Tower calibration ZDCTowerParam for timestamp " << mgr.getTimestamp();
        towerParam->print();
      }
    }

    LOG(info) << loadedConfFiles;

    mDR.setModuleConfig(moduleConfig);
    mDR.setRecoConfigZDC(recoConfigZDC);
    mDR.setTDCParam(tdcParam);
    mDR.setTDCCorr(tdcCorr);
    mDR.setEnergyParam(energyParam);
    mDR.setTowerParam(towerParam);

    if (mDebugOut) {
      mDR.setDebugOutput();
    }

    mDR.setVerbosity(mVerbosity);

    mDR.init();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcdata = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chans = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  mDR.process(peds, bcdata, chans);
  const std::vector<o2::zdc::RecEventAux>& recAux = mDR.getReco();

  // Transfer wafeform
  bool fullinter = mDR.getFullInterpolation();
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
  mDR.eor();
  LOGF(info, "ZDC Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getDigitRecoSpec(const int verbosity = 0, const bool enableDebugOut = false)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);

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
    o2::framework::Options{{"ccdb-url", o2::framework::VariantType::String, o2::base::NameConf::getCCDBServer(), {"CCDB Url"}}}};
}

} // namespace zdc
} // namespace o2
