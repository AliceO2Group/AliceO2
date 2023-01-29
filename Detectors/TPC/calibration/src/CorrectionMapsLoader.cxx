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

#include "TPCCalibration/CorrectionMapsLoader.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCBase/CDBInterface.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ConcreteDataMatcher.h"

using namespace o2::tpc;
using namespace o2::framework;

#ifndef GPUCA_GPUCODE_DEVICE

//________________________________________________________
void CorrectionMapsLoader::updateVDrift(float vdriftCorr, float vdrifRef, float driftTimeOffset)
{
  o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCorrMap, 0, vdriftCorr, vdrifRef, driftTimeOffset);
  if (mCorrMapRef) {
    o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCorrMapRef, 0, vdriftCorr, vdrifRef, driftTimeOffset);
  }
}

//________________________________________________________
void CorrectionMapsLoader::extractCCDBInputs(ProcessingContext& pc)
{
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMap");
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMapRef"); // not used at the moment
}

//________________________________________________________
void CorrectionMapsLoader::requestCCDBInputs(std::vector<InputSpec>& inputs)
{
  addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});          // time-dependent
  addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
}

//________________________________________________________
void CorrectionMapsLoader::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//________________________________________________________
bool CorrectionMapsLoader::accountCCDBInputs(const ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("TPC", "CorrMap", 0)) {
    setCorrMap((o2::gpu::TPCFastTransform*)obj);
    mCorrMap->rectifyAfterReadingFromFile();
    setUpdatedMap();
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapRef", 0)) {
    setCorrMapRef((o2::gpu::TPCFastTransform*)obj);
    mCorrMapRef->rectifyAfterReadingFromFile();
    setUpdatedMapRef();
    return true;
  }
  return false;
}
#endif // #ifndef GPUCA_GPUCODE_DEVICE
