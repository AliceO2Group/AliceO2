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

#include "TPCCalibration/CorrectionMapsHelper.h"
#include "TPCBase/CDBInterface.h"
#include "TPCFastTransform.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ConcreteDataMatcher.h"

using namespace o2::tpc;
using namespace o2::framework;

//________________________________________________________
void CorrectionMapsHelper::extractCCDBInputs(ProcessingContext& pc)
{
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMap");
  // pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMapRef"); // not used at the moment
}

//________________________________________________________
void CorrectionMapsHelper::requestCCDBInputs(std::vector<InputSpec>& inputs)
{
  addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)}); // time-dependent
  //  addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
}

//________________________________________________________
void CorrectionMapsHelper::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//________________________________________________________
bool CorrectionMapsHelper::accountCCDBInputs(const ConcreteDataMatcher& matcher, void* obj)
{
  /*
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapRef", 0)) {
    mCorrMapRef.reset((o2::gpu::TPCFastTransform*)obj);
    mCorrMapRef->rectifyAfterReadingFromFile();
    mUpdated = true;
    return true;
  }
  */
  if (matcher == ConcreteDataMatcher("TPC", "CorrMap", 0)) {
    mCorrMap.reset((o2::gpu::TPCFastTransform*)obj);
    mCorrMap->rectifyAfterReadingFromFile();
    mUpdated = true;
    return true;
  }
  return false;
}
