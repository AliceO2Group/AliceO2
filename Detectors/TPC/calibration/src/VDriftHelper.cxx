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

#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "DataFormatsTPC/LtrCalibData.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/TimingInfo.h"

using namespace o2::tpc;
using namespace o2::framework;

//________________________________________________________
VDriftHelper::VDriftHelper()
{
  const auto& gaspar = o2::tpc::ParameterGas::Instance();
  const auto& detpar = o2::tpc::ParameterDetector::Instance();
  const auto& elpar = o2::tpc::ParameterElectronics::Instance();
  mVD.corrFact = 1.0;
  mVD.refVDrift = gaspar.DriftV;
  mVD.refTimeOffset = detpar.DriftTimeOffset * elpar.ZbinWidth; // convert time bins to \mus
  // was it imposed from the command line?
  mVD.creationTime = 1;                                                                                                         // just to be above 0
  if (o2::conf::ConfigurableParam::getProvenance("TPCGasParam.DriftV") == o2::conf::ConfigurableParam::EParamProvenance::kRT) { // we stick to this value
    mVD.creationTime = std::numeric_limits<long>::max();
    mForceParamDrift = true;
    LOGP(info, "TPC VDrift was set from command line to {}, will neglect update from CCDB", mVD.refVDrift);
  }
  if (o2::conf::ConfigurableParam::getProvenance("TPCDetParam.DriftTimeOffset") == o2::conf::ConfigurableParam::EParamProvenance::kRT) { // we stick to this value
    mVD.creationTime = std::numeric_limits<long>::max();
    mForceParamOffset = true;
    LOGP(info, "TPC drift time offset was set from command line to {} mus ({} TB), will neglect update from CCDB",
         mVD.refTimeOffset, detpar.DriftTimeOffset);
  }

  // check if temperature and pressure is set from the command line
  if ((o2::conf::ConfigurableParam::getProvenance("TPCGasParam.Temperature") == o2::conf::ConfigurableParam::EParamProvenance::kRT) && (o2::conf::ConfigurableParam::getProvenance("TPCGasParam.Pressure") == o2::conf::ConfigurableParam::EParamProvenance::kRT)) { // we stick to this value
    mForceTPScaling = true;
    LOGP(info, "VDriftHelper: Temperature and pressure were set from command line to {} C and {} mbar, will neglect updates from CCDB", gaspar.Temperature, gaspar.Pressure);
    if (gaspar.Temperature <= 0 || gaspar.Pressure <= 0) {
      LOGP(info, "VDriftHelper: Disabling VDrift scaling with T / P");
    }
  }

  mUpdated = true;
  mSource = Source::Param;
}

//________________________________________________________
void VDriftHelper::accountLaserCalibration(const LtrCalibData* calib, long fallBackTimeStamp)
{
  if (!calib || mForceParamDrift) { // laser may set only DriftParam (the offset is 0)
    return;
  }
  if (!calib->isValid()) {
    LOGP(warn, "Ignoring invalid laser calibration (corrections: A-side={}, C-side={}, NTracks: A-side={} C-side={})", calib->dvCorrectionA, calib->dvCorrectionC, calib->nTracksA, calib->nTracksC);
    return;
  }
  // old entries of laser calib have no update time assigned
  long updateTS = calib->creationTime > 0 ? calib->creationTime : fallBackTimeStamp;
  LOG(info) << "accountLaserCalibration " << calib->refVDrift << " / " << calib->getDriftVCorrection() << " t " << updateTS << " vs " << mVDLaser.creationTime;
  // old entries of laser calib have no reference assigned
  float ref = calib->refVDrift > 0. ? calib->refVDrift : o2::tpc::ParameterGas::Instance().DriftV;
  float corr = calib->getDriftVCorrection();
  if (corr > 0.) { // laser correction is inverse multiplicative
    static bool firstCall = true;
    auto prevRef = mVDLaser.refVDrift;
    mVDLaser.refVDrift = ref;
    mVDLaser.corrFact = 1. / corr;
    mVDLaser.creationTime = calib->creationTime;
    mVDLaser.refTimeOffset = calib->refTimeOffset;
    mVDLaser.refTP = calib->tp;
    mUpdated = true;
    mSource = Source::Laser;
    if (mMayRenormSrc & (0x1U << Source::Laser)) { // this was 1st setting?
      if (corr != 1.f) {                           // this may happen if old-style (non-normalized) standalone or non-normalized run-time laset calibration is used
        LOGP(warn, "VDriftHelper: renorming initial TPC refVDrift={}/correction={} to {}/1.0, source: {}", mVDLaser.refVDrift, mVDLaser.corrFact, mVDLaser.getVDrift(), getSourceName(mSource));
        mVDLaser.normalize(); // renorm reference to have correction = 1.
      }
      mMayRenormSrc &= ~(0x1U << Source::Laser); // unset MayRenorm
    } else if (ref != prevRef) {                 // we want to keep the same reference over the run, this may happen if run-time laser calibration is supplied
      LOGP(warn, "VDriftHelper: renorming updated TPC refVDrift={}/correction={} previous refVDrift {}, source: {}", mVDLaser.refVDrift, mVDLaser.corrFact, prevRef, getSourceName(mSource));
      mVDLaser.normalize(prevRef);
    }
  }
}

//________________________________________________________
void VDriftHelper::accountDriftCorrectionITSTPCTgl(const VDriftCorrFact* calib)
{
  if (!calib || (mForceParamDrift && mForceParamOffset)) {
    return;
  }
  LOG(info) << "accountDriftCorrectionITSTPCTgl " << calib->corrFact << " t " << calib->creationTime << " vs " << mVDTPCITSTgl.creationTime;
  auto prevRefVDrift = mVDTPCITSTgl.refVDrift;
  auto prevRefTOffs = mVDTPCITSTgl.refTimeOffset;
  mVDTPCITSTgl = *calib;
  mUpdated = true;
  mSource = Source::ITSTPCTgl;
  if (mMayRenormSrc & (0x1U << Source::ITSTPCTgl)) {         // this was 1st setting?
    if (!mForceParamDrift && mVDTPCITSTgl.corrFact != 1.f) { // this may happen if calibration from prevous run is used
      LOGP(warn, "VDriftHelper: renorming initial TPC refVDrift={}/correction={} to {}/1.0, source: {}", mVDTPCITSTgl.refVDrift, mVDTPCITSTgl.corrFact, mVDTPCITSTgl.getVDrift(), getSourceName(mSource));
      mVDTPCITSTgl.normalize(); // renorm reference to have correction = 1.
    }
    if (!mForceParamOffset && mVDTPCITSTgl.timeOffsetCorr != 0.) {
      LOGP(warn, "VDriftHelper: renorming initial TPC refTimeOffset={}/correction={} to {}/0.0, source: {}", mVDTPCITSTgl.refTimeOffset, mVDTPCITSTgl.timeOffsetCorr, mVDTPCITSTgl.getTimeOffset(), getSourceName());
      mVDTPCITSTgl.normalizeOffset();
    }
    mMayRenormSrc &= ~(0x1U << Source::ITSTPCTgl); // unset MayRenorm
  } else {
    if (!mForceParamDrift && mVDTPCITSTgl.refVDrift != prevRefVDrift) { // we want to keep the same reference over the run, this should not happen!
      LOGP(warn, "VDriftHelper: renorming updated TPC refVDrift={}/correction={} previous refVDrift {}, source: {}", mVDTPCITSTgl.refVDrift, mVDTPCITSTgl.corrFact, prevRefVDrift, getSourceName());
      mVDTPCITSTgl.normalize(prevRefVDrift);
    }
    if (!mForceParamOffset && mVDTPCITSTgl.refTimeOffset != prevRefTOffs) { // we want to keep the same reference over the run, this should not happen!
      LOGP(warn, "VDriftHelper: renorming updated TPC refTimeOffset={}/correction={} previous refTimeOffset {}, source: {}", mVDTPCITSTgl.refTimeOffset, mVDTPCITSTgl.timeOffsetCorr, prevRefTOffs, getSourceName());
      mVDTPCITSTgl.normalizeOffset(prevRefTOffs);
    }
  }
}

//________________________________________________________
void VDriftHelper::extractCCDBInputs(ProcessingContext& pc, bool laser, bool itstpcTgl)
{
  if (mForceParamDrift && mForceParamOffset) { // fixed from the command line
    return;
  }
  if (laser && !mForceParamDrift) {
    pc.inputs().get<o2::tpc::LtrCalibData*>("laserCalib");
  }
  if (itstpcTgl) {
    pc.inputs().get<o2::tpc::VDriftCorrFact*>("vdriftTgl");
  }
  mPTHelper.extractCCDBInputs(pc);

  if (mUpdated || mIsTPScalingPossible) { // there was a change
    // prefer among laser and tgl VDrift the one with the latest update time
    auto saveVD = mVD;

    // apply TP scaling of mVD if possible
    if (float tp = mPTHelper.getTP(pc.services().get<o2::framework::TimingInfo>().creation); tp > 0) {
      // try to extract refTP if needed
      auto& vd = (mVDTPCITSTgl.creationTime < mVDLaser.creationTime) ? mVDLaser : mVDTPCITSTgl;
      if (mForceTPScaling) {
        const auto& gaspar = o2::tpc::ParameterGas::Instance();
        tp = (gaspar.Temperature > 0 && gaspar.Pressure > 0) ? ((gaspar.Temperature + 273.15) / gaspar.Pressure) : -1;
        mIsTPScalingPossible = (tp > 0) && (vd.refTP > 0 || extractTPForVDrift(vd));
      } else {
        mIsTPScalingPossible = (vd.refTP > 0) || extractTPForVDrift(vd);
      }
      if (mIsTPScalingPossible) {
        // if no new VDrift object was loaded and if delta TP is small, do not rescale and return
        if (!mUpdated && std::abs(tp - vd.refTP) < 1e-5) {
          return;
        }
        mUpdated = true;
        vd.normalize(0, tp);
        if (vd.creationTime == saveVD.creationTime) {
          LOGP(info, "VDriftHelper: Scaling VDrift from {} to {} with T/P from {} to {}", saveVD.getVDrift(), vd.getVDrift(), saveVD.refTP, vd.refTP);
        } else {
          LOGP(info, "VDriftHelper: Init new VDrift of {} with T/P {}", vd.getVDrift(), vd.refTP);
        }
      }
    }

    mVD = mVDTPCITSTgl.creationTime < mVDLaser.creationTime ? mVDLaser : mVDTPCITSTgl;
    auto& loserVD = mVDTPCITSTgl.creationTime < mVDLaser.creationTime ? mVDTPCITSTgl : mVDLaser;

    if (mForceParamDrift) {
      mVD.refVDrift = saveVD.refVDrift;
      mVD.corrFact = saveVD.corrFact;
      mVD.corrFactErr = 0.f;
    }
    if (mForceParamOffset) {
      mVD.refTimeOffset = saveVD.refTimeOffset;
      mVD.timeOffsetCorr = 0.f;
    }
    mSource = mVDTPCITSTgl.creationTime < mVDLaser.creationTime ? Source::Laser : Source::ITSTPCTgl;
    auto loseCTime = loserVD.creationTime;
    loserVD = mVD; // override alternative VD to avoid normalization problems later
    loserVD.creationTime = loseCTime;
    std::string rep = fmt::format("Prefer TPC Drift from {} with time {} to {} with time {}",
                                  SourceNames[int(mSource)], mVD.creationTime, mSource == Source::Laser ? SourceNames[int(Source::ITSTPCTgl)] : SourceNames[int(Source::Laser)],
                                  mSource == Source::Laser ? mVDTPCITSTgl.creationTime : mVDLaser.creationTime);
    if (mForceParamDrift || mForceParamOffset) {
      std::string impos = mForceParamDrift ? "VDrift" : "";
      if (mForceParamOffset) {
        impos += mForceParamDrift ? " and DriftTimeOffset" : "DriftTimeOffset";
      }
      rep += fmt::format(" but {} imposed from command line", impos);
    }
    LOGP(info, "{}", rep);
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
  // adding pressure and temperature inputs
  PressureTemperatureHelper::requestCCDBInputs(inputs);
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
  return mPTHelper.accountCCDBInputs(matcher, obj);
}

bool VDriftHelper::extractTPForVDrift(VDriftCorrFact& vdrift, int64_t tsStepMS)
{
  const int64_t tsStart = vdrift.firstTime;
  const int64_t tsEnd = vdrift.lastTime;

  if (tsStart == tsEnd) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      LOGP(warn, "VDriftHelper: Cannot extract T/P for VDrift with identical start/end time {}!", tsStart);
    }
    return false;
  }

  // make sanity check of the time range
  const auto [minValidTime, maxValidTime] = mPTHelper.getMinMaxTime();
  const int64_t minTimeAccepted = static_cast<int64_t>(minValidTime) - 20 * o2::ccdb::CcdbObjectInfo::MINUTE;
  const int64_t maxTimeAccepted = static_cast<int64_t>(maxValidTime) + 20 * o2::ccdb::CcdbObjectInfo::MINUTE;

  // check if the stored time stamp range is valid i.e. check if the range is in the vicinity of the current time
  if ((minTimeAccepted > tsEnd) || (tsStart > maxTimeAccepted)) {
    // check if creation time can be used
    LOGP(warn, "VDriftHelper: Time range of VDrift object {} - {} is not valid for time range of T/P object {} - {}! Do not extract ref. T/P for VDrift!", tsStart, tsEnd, minValidTime, maxValidTime);
    return false;
  }

  double meanTP = 0;
  int countTP = 0;

  for (int64_t ts = tsStart; ts < tsEnd; ts += tsStepMS) {
    meanTP += mPTHelper.getTP(ts);
    ++countTP;
  }

  if (countTP == 0) {
    LOGP(error, "VDriftHelper: Could not get T/P for time range {} -> {}", tsStart, tsEnd);
    return false;
  }

  meanTP /= countTP;

  LOGP(info, "VDriftHelper: Setting mean T/P for VDrift to {} for time range {} -> {}", meanTP, tsStart, tsEnd);
  vdrift.refTP = meanTP;
  return true;
}
