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

#include <TMath.h>
#include "Framework/Logger.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCReconstruction/DigiRecoTest.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"
#include "ZDCReconstruction/ZDCTowerParam.h"

namespace o2
{
namespace zdc
{

//______________________________________________________________________________
void DigiRecoTest::init()
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  if (mCCDBServer.size() > 0) {
    mgr.setURL(mCCDBServer);
    mDigi.setCCDBServer(mCCDBServer);
  }
  mDigi.init();

  LOG(info) << "Initialization of ZDC Test reconstruction " << mgr.getURL();
  // CCDB server has been initialized in DigitizerTest
  long timeStamp = mgr.getTimestamp();

  auto* moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
  if (!moduleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object @ " << o2::zdc::CCDBPathConfigModule;
    return;
  }
  LOG(info) << "Loaded ModuleConfig for timestamp " << timeStamp;
  moduleConfig->print();

  // Configuration parameters for ZDC reconstruction
  auto* recoConfigZDC = mgr.get<o2::zdc::RecoConfigZDC>(o2::zdc::CCDBPathRecoConfigZDC);
  if (!recoConfigZDC) {
    LOG(fatal) << "Missing RecoConfigZDC object";
    return;
  }
  LOG(info) << "Loaded RecoConfigZDC for timestamp " << timeStamp;
  recoConfigZDC->print();

  // TDC centering
  auto* tdcParam = mgr.get<o2::zdc::ZDCTDCParam>(o2::zdc::CCDBPathTDCCalib);
  if (!tdcParam) {
    LOG(fatal) << "Missing ZDCTDCParam calibration object";
    return;
  }
  LOG(info) << "Loaded TDC centering ZDCTDCParam for timestamp " << timeStamp;
  tdcParam->print();

  // TDC correction parameters
  auto* tdcCorr = mgr.get<o2::zdc::ZDCTDCCorr>(o2::zdc::CCDBPathTDCCorr);
  if (!tdcCorr) {
    LOG(fatal) << "Missing ZDCTDCCorr calibration object - no correction is applied";
  } else {
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded TDC correction parameters for timestamp " << timeStamp;
      tdcCorr->print();
    }
  }

  // Energy calibration
  auto* energyParam = mgr.get<o2::zdc::ZDCEnergyParam>(o2::zdc::CCDBPathEnergyCalib);
  if (!energyParam) {
    LOG(warning)
      << "Missing ZDCEnergyParam calibration object - using default";
  } else {
    LOG(info) << "Loaded Energy calibration ZDCEnergyParam for timestamp " << timeStamp;
    energyParam->print();
  }

  // Tower calibration
  auto* towerParam =
    mgr.get<o2::zdc::ZDCTowerParam>(o2::zdc::CCDBPathTowerCalib);
  if (!towerParam) {
    LOG(warning)
      << "Missing ZDCTowerParam calibration object - using default";
  } else {
    LOG(info) << "Loaded Tower calibration ZDCTowerParam for timestamp " << timeStamp;
    towerParam->print();
  }

  mDR.setModuleConfig(moduleConfig);
  mDR.setRecoConfigZDC(recoConfigZDC);
  mDR.setTDCParam(tdcParam);
  mDR.setTDCCorr(tdcCorr);
  mDR.setEnergyParam(energyParam);
  mDR.setTowerParam(towerParam);
  mDR.setVerbosity(mVerbosity);
  mDR.init();

  const uint32_t* mask = mDR.getChMask();
  for (uint32_t ic = 0; ic < NChannels; ic++) {
    mDigi.setMask(ic, mask[ic]);
  }
} // init

} // namespace zdc
} // namespace o2
