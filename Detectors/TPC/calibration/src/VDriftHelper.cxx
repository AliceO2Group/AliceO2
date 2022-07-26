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

#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "DataFormatsTPC/LtrCalibData.h"
#include "TPCBase/ParameterGas.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ConcreteDataMatcher.h"

using namespace o2::tpc;
using namespace o2::framework;

//________________________________________________________
VDriftHelper::VDriftHelper()
{
  const auto& gaspar = o2::tpc::ParameterGas::Instance();
  mVD.corrFact = 1.0;
  mVD.refVDrift = gaspar.DriftV;
  // was it imposed from the command line?
  if (o2::conf::ConfigurableParam::getProvenance("TPCGasParam.DriftV") == o2::conf::ConfigurableParam::EParamProvenance::kRT) { // we stick to this value
    mVD.creationTime = std::numeric_limits<long>::max();
    LOGP(info, "TPC VDrift was set from command line to {}, will neglect update from CCDB", mVD.refVDrift);
  } else {
    mVD.creationTime = 1; // just to be above 0
  }
  mUpdated = true;
  mSource = Source::GasParam;
}

//________________________________________________________
void VDriftHelper::accountLaserCalibration(const LtrCalibData* calib, long fallBackTimeStamp)
{
  if (!calib) {
    return;
  }
  // old entries of laser calib have no update time assigned
  long updateTS = calib->creationTime > 0 ? calib->creationTime : fallBackTimeStamp;
  LOG(info) << "accountLaserCalibration " << calib->getDriftVCorrection() << " t " << updateTS << " vs " << mVD.creationTime;
  if (updateTS < mVD.creationTime) { // prefer current value
    return;
  }
  // old entries of laser calib have no reference assigned
  float ref = calib->refVDrift > 0. ? calib->refVDrift : o2::tpc::ParameterGas::Instance().DriftV;
  float corr = calib->getDriftVCorrection();
  if (corr > 0.) { // laser correction is inverse multiplicative
    static bool firstCall = true;
    auto prevRef = mVD.refVDrift;
    mVD.refVDrift = ref;
    mVD.corrFact = 1. / corr;
    mUpdated = true;
    mSource = Source::Laser;
    if (mMayRenorm) {    // this was 1st setting?
      if (corr != 1.f) { // this may happen if old-style (non-normalized) standalone or non-normalized run-time laset calibration is used
        LOGP(warn, "VDriftHelper: renorming initinal TPC refVDrift={}/correction={} to {}/1.0, source: {}", mVD.refVDrift, mVD.corrFact, mVD.getVDrift(), getSourceName());
        mVD.normalize(); // renorm reference to have correction = 1.
      }
      mMayRenorm = false;
    } else if (ref != prevRef) { // we want to keep the same reference over the run, this may happen if run-time laser calibration is supplied
      LOGP(warn, "VDriftHelper: renorming updated TPC refVDrift={}/correction={} previous refVDrift {}, source: {}", mVD.refVDrift, mVD.corrFact, prevRef, getSourceName());
      mVD.normalize(prevRef);
    }
  }
}

//________________________________________________________
void VDriftHelper::accountDriftCorrectionITSTPCTgl(const VDriftCorrFact* calib)
{
  LOG(info) << "accountDriftCorrectionITSTPCTgl " << calib->corrFact << " t " << calib->creationTime << " vs " << mVD.creationTime;
  if (!calib || calib->creationTime < mVD.creationTime) { // prefer current value
    return;
  }
  auto prevRef = mVD.refVDrift;
  mVD = *calib;
  mUpdated = true;
  mSource = Source::ITSTPCTgl;
  if (mMayRenorm) {            // this was 1st setting?
    if (mVD.corrFact != 1.f) { // this may happen if calibration from prevous run is used
      LOGP(warn, "VDriftHelper: renorming initinal TPC refVDrift={}/correction={} to {}/1.0, source: {}", mVD.refVDrift, mVD.corrFact, mVD.getVDrift(), getSourceName());
      mVD.normalize(); // renorm reference to have correction = 1.
    }
    mMayRenorm = false;
  } else if (mVD.refVDrift != prevRef) { // we want to keep the same reference over the run, this should not happen!
    LOGP(alarm, "VDriftHelper: renorming updated TPC refVDrift={}/correction={} previous refVDrift {}, source: {}", mVD.refVDrift, mVD.corrFact, prevRef, getSourceName());
    mVD.normalize(prevRef);
  }
}

//________________________________________________________
void VDriftHelper::extractCCDBInputs(ProcessingContext& pc, bool laser, bool itstpcTgl)
{
  if (laser) {
    pc.inputs().get<o2::tpc::LtrCalibData*>("laserCalib");
  }
  if (itstpcTgl) {
    pc.inputs().get<o2::tpc::VDriftCorrFact*>("vdriftTgl");
  }
}

//________________________________________________________
void VDriftHelper::requestCCDBInputs(std::vector<InputSpec>& inputs, bool laser, bool itstpcTgl)
{
  if (laser) {
    addInput(inputs, {"laserCalib", "TPC", "CalibLaserTracks", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalLaserTracks))});
  }
  if (itstpcTgl) {
    // VDrift calibration may change during the run (in opposite to Laser calibration, at least at the moment), so ask per-TF query
    addInput(inputs, {"vdriftTgl", "TPC", "VDriftTgl", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalVDriftTgl), {}, 1)});
  }
}

//________________________________________________________
void VDriftHelper::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//________________________________________________________
bool VDriftHelper::accountCCDBInputs(const ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("TPC", "VDriftTgl", 0)) {
    accountDriftCorrectionITSTPCTgl(static_cast<VDriftCorrFact*>(obj));
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CalibLaserTracks", 0)) {
    accountLaserCalibration(static_cast<LtrCalibData*>(obj));
    return true;
  }
  return false;
}
