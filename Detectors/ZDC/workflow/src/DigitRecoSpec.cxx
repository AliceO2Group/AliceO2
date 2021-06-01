// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitRecoSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <vector>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ZDCWorkflow/DigitRecoSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "FairLogger.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/ZDCIntegrationParam.h"
#include "ZDCReconstruction/ZDCTDCParam.h"

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

void DigitRecoSpec::init(o2::framework::InitContext& ic)
{
  // At this stage we cannot access the CCDB yet
}

void DigitRecoSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    // Initialization from CCDB
    std::string ccdbHost = "http://ccdb-test.cern.ch:8080";

    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(ccdbHost);
    long timeStamp = 0;
    if (timeStamp == mgr.getTimestamp()) {
      return;
    }
    mgr.setTimestamp(timeStamp);
    auto* moduleConfig =
      mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(FATAL) << "Missing configuration object";
      return;
    }
    LOG(INFO) << "Loaded module configuration for timestamp " << timeStamp;

    // Get Reconstruction parameters
    auto* integrationParam =
      mgr.get<o2::zdc::ZDCIntegrationParam>(o2::zdc::CCDBPathConfigIntegration);
    if (!integrationParam) {
      LOG(FATAL) << "Missing ZDCIntegrationParam object";
      return;
    }
    mDR.setModuleConfig(moduleConfig);
    mDR.setIntegrationParam(integrationParam);
    mDR.setDebugOutput();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcdata = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chans = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  mDR.process(peds, bcdata, chans);
  const std::vector<o2::zdc::RecEventAux>& recAux = mDR.getReco();

  RecEvent recEvent;

  LOG(INFO) << "BC in recAux " << recAux.size();
  for (auto reca : recAux) {
    int32_t ne = reca.ezdc.size();
    int32_t nt = 0;
    for (int32_t it = 0; it < o2::zdc::NTDCChannels; it++) {
      for (int32_t ih = 0; ih < reca.ntdc[it]; ih++) {
        if (nt == 0) {
          recEvent.addBC(reca.ir, reca.channels, reca.triggers);
        }
        nt++;
        recEvent.addTDC(it, reca.tdcVal[it][ih], reca.tdcAmp[it][ih]);
      }
    }
    if (ne > 0 && nt == 0) {
      recEvent.addBC(reca.ir);
    }
    if (ne > 0) {
      std::map<uint8_t, float>::iterator it;
      for (it = reca.ezdc.begin(); it != reca.ezdc.end(); it++) {
        recEvent.addEnergy(it->first, it->second);
      }
    }
    if (nt > 0 || ne > 0) {
      printf("Orbit %9u bc %4u ntdc %2d ne %2d\n", reca.ir.orbit, reca.ir.bc, nt, ne);
    }
  }

  pc.outputs().snapshot(Output{"ZDC", "BCREC", 0, Lifetime::Timeframe}, recEvent.mRecBC);
  pc.outputs().snapshot(Output{"ZDC", "ENERGY", 0, Lifetime::Timeframe}, recEvent.mEnergy);
  pc.outputs().snapshot(Output{"ZDC", "TDCDATA", 0, Lifetime::Timeframe}, recEvent.mTDCData);
  mTimer.Stop();
  LOG(INFO) << "Reconstructed ZDC data for " << recEvent.mRecBC.size() << " b.c. in " << mTimer.CpuTime() - cput << " s";
}

void DigitRecoSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "ZDC Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getDigitRecoSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "BCREC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "ENERGY", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "TDCDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "zdc-digi-reco",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DigitRecoSpec>()}};
}

} // namespace zdc
} // namespace o2
