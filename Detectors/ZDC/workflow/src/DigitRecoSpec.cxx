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
#include <cstdlib>
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

DigitRecoSpec::DigitRecoSpec(const int verbosity, const bool debugOut,
                             const bool enableZDCTDCCorr, const bool enableZDCEnergyParam, const bool enableZDCTowerParam, const bool enableBaselineParam)
  : mVerbosity(verbosity), mDebugOut(debugOut), mEnableZDCTDCCorr(enableZDCTDCCorr), mEnableZDCEnergyParam(enableZDCEnergyParam), mEnableZDCTowerParam(enableZDCTowerParam), mEnableBaselineParam(enableBaselineParam)
{
  mTimer.Stop();
  mTimer.Reset();
}

void DigitRecoSpec::init(o2::framework::InitContext& ic)
{
  mMaxWave = ic.options().get<int>("max-wave");
  if (mMaxWave > 0) {
    LOG(warning) << "Limiting the number of waveforms in ourput to " << mMaxWave;
  }
  mRecoFraction = ic.options().get<double>("tf-fraction");
  if (mRecoFraction < 0 || mRecoFraction > 1) {
    LOG(error) << "Unphysical reconstructed fraction " << mRecoFraction << " set to 1.0";
    mRecoFraction = 1.0;
  }
  if (mRecoFraction < 1) {
    LOG(warning) << "Target fraction for reconstructed TFs = " << mRecoFraction;
  }
}

void DigitRecoSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
  pc.inputs().get<o2::zdc::RecoConfigZDC*>("recoconfig");
  pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
  if (mEnableZDCTDCCorr) {
    pc.inputs().get<o2::zdc::ZDCTDCCorr*>("tdccorr");
  }
  if (mEnableZDCEnergyParam) {
    pc.inputs().get<o2::zdc::ZDCEnergyParam*>("adccalib");
  }
  if (mEnableZDCTowerParam) {
    pc.inputs().get<o2::zdc::ZDCTowerParam*>("towercalib");
  }
  if (mEnableBaselineParam) {
    pc.inputs().get<o2::zdc::BaselineParam*>("basecalib");
  }
}

void DigitRecoSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    auto* config = (const o2::zdc::ModuleConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setModuleConfig(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "RECOCONFIG", 0)) {
    // Configuration parameters for ZDC reconstruction
    auto* config = (const o2::zdc::RecoConfigZDC*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setRecoConfigZDC(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIB", 0)) {
    // TDC centering
    auto* config = (const o2::zdc::ZDCTDCParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCORR", 0)) {
    // TDC correction parameters
    auto* config = (const o2::zdc::ZDCTDCCorr*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCCorr(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "ADCCALIB", 0)) {
    // Energy calibration
    auto* config = (const o2::zdc::ZDCEnergyParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setEnergyParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TOWERCALIB", 0)) {
    // Tower intercalibration
    auto* config = (const o2::zdc::ZDCTowerParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTowerParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "BASECALIB", 0)) {
    // Average pedestals
    auto* config = (const o2::zdc::BaselineParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setBaselineParam(config);
  }
}

void DigitRecoSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    LOG(info) << "DigitRecoSpec::run initialization";
    mInitialized = true;
    updateTimeDependentParams(pc);
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

  // Reduce load by dropping random time frames
  bool toProcess = true;
  if (mRecoFraction <= 0.) {
    toProcess = false;
  } else if (mRecoFraction < 1.0) {
    double frac = std::rand() / double(RAND_MAX);
    if (frac > mRecoFraction) {
      toProcess = false;
    }
  }

  RecEvent recEvent;

  if (toProcess) {
    int rval = mWorker.process(peds, bcdata, chans);
    if (rval != 0 || mWorker.inError()) {
      LOG(warning) << bcdata.size() << " BC " << chans.size() << " CH " << peds.size() << " OD -> processing ended in ERROR @ line " << rval;
    } else {
      const std::vector<o2::zdc::RecEventAux>& recAux = mWorker.getReco();

      // Transfer wafeform
      bool fullinter = mWorker.getFullInterpolation();
      if (mVerbosity > 0 || fullinter) {
        LOGF(info, "BC processed during reconstruction %d%s", recAux.size(), (fullinter ? " FullInterpolation" : ""));
      }
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
          // Limit the number of waveforms in output message
          if (mMaxWave > 0 && ntw >= mMaxWave) {
            if (mVerbosity > DbgMinimal) {
              LOG(warning) << "Maximum number of output waveforms per TF reached: " << mMaxWave;
            }
            break;
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
      LOG(info) << bcdata.size() << " BC " << chans.size() << " CH " << peds.size() << " OD "
                << "-> Reconstructed " << ntt << " signal TDCs and " << nte << " ZDC energies and "
                << nti << " info messages in " << recEvent.mRecBC.size() << "/" << recAux.size() << " b.c. and "
                << ntw << " waveform chunks";
    }
  } else {
    LOG(info) << bcdata.size() << " BC " << chans.size() << " CH " << peds.size() << " OD "
              << "-> SKIPPED because of requested reconstruction fraction = " << mRecoFraction;
  }
  // TODO: rate information for all channels
  // TODO: summary of reconstruction to be collected by DQM?
  pc.outputs().snapshot(Output{"ZDC", "BCREC", 0}, recEvent.mRecBC);
  pc.outputs().snapshot(Output{"ZDC", "ENERGY", 0}, recEvent.mEnergy);
  pc.outputs().snapshot(Output{"ZDC", "TDCDATA", 0}, recEvent.mTDCData);
  pc.outputs().snapshot(Output{"ZDC", "INFO", 0}, recEvent.mInfo);
  pc.outputs().snapshot(Output{"ZDC", "WAVE", 0}, recEvent.mWaveform);
  mTimer.Stop();
}

void DigitRecoSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.eor();
  LOGF(info, "ZDC Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getDigitRecoSpec(const int verbosity = 0, const bool enableDebugOut = true,
                                              const bool enableZDCTDCCorr = true, const bool enableZDCEnergyParam = true, const bool enableZDCTowerParam = true, const bool enableBaselineParam = true)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));
  inputs.emplace_back("recoconfig", "ZDC", "RECOCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathRecoConfigZDC.data()));
  inputs.emplace_back("tdccalib", "ZDC", "TDCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathTDCCalib.data()));
  if (enableZDCTDCCorr) {
    inputs.emplace_back("tdccorr", "ZDC", "TDCCORR", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathTDCCorr.data()));
  } else {
    LOG(warning) << "ZDCTDCCorr has been disabled - no correction is applied";
  }
  if (enableZDCEnergyParam) {
    inputs.emplace_back("adccalib", "ZDC", "ADCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathEnergyCalib.data()));
  } else {
    LOG(warning) << "ZDCEnergyParam has been disabled - no energy calibration is applied";
  }
  if (enableZDCTowerParam) {
    inputs.emplace_back("towercalib", "ZDC", "TOWERCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathTowerCalib.data()));
  } else {
    LOG(warning) << "ZDCTowerParam has been disabled - no tower intercalibration";
  }
  if (enableBaselineParam) {
    inputs.emplace_back("basecalib", "ZDC", "BASECALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathBaselineCalib.data()));
  } else {
    LOG(warning) << "BaselineParam has been disabled - no fallback in case orbit pedestals are missing";
  }

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
    AlgorithmSpec{adaptFromTask<DigitRecoSpec>(verbosity, enableDebugOut, enableZDCTDCCorr, enableZDCEnergyParam, enableZDCTowerParam, enableBaselineParam)},
    o2::framework::Options{{"max-wave", o2::framework::VariantType::Int, 0, {"Maximum number of waveforms per TF in output"}},
                           {"tf-fraction", o2::framework::VariantType::Double, 1.0, {"Fraction of reconstructed TFs"}}}};
}

} // namespace zdc
} // namespace o2
