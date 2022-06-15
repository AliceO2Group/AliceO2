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

#include <GRPCalibration/GRPDCSDPsProcessor.h>
#include "DetectorsCalibration/Utils.h"
#include "Rtypes.h"
#include <cmath>

using namespace o2::grp;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

//__________________________________________________________________

void GRPDCSDPsProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by GRP
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
  }
  mMagFieldHelper.verbose = mVerbose;
}
//__________________________________________________________________

int GRPDCSDPsProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerbose) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }
  if (!mFirstTimeSet) {
    mFirstTime = mStartValidity;
    mFirstTimeSet = true;
  }
  mMagFieldHelper.updated = false;

  mUpdateLHCIFInfo = false;   // by default, we do not foresee a new entry in the CCDB for the LHCIF DPs

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    processDP(it);
    mPids[it.id] = true;
  }

  if (isMagFieldUpdated()) {
    updateMagFieldCCDB();
  }
  mCallSlice++;
  return 0;
}

//__________________________________________________________________
int GRPDCSDPsProcessor::processDP(const DPCOM& dpcom)
{

  // processing single DP

  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  auto& val = dpcom.data;
  if (mVerbose) {
    if (type == DPVAL_DOUBLE) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with double value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_BOOL) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with bool value = " << o2::dcs::getValue<bool>(dpcom);
    } else if (type == DPVAL_STRING) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with string value = " << o2::dcs::getValue<std::string>(dpcom);
    }
  }
  auto flags = val.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    // now I need to access the correct element
    std::string aliasStr(dpid.get_alias());
    // LOG(info) << "alias 0 = " << aliasStr;
    //  B-field DPs
    if (aliasStr == "L3Current") {
      mMagFieldHelper.updateCurL3(static_cast<float>(o2::dcs::getValue<double>(dpcom)));
    } else if (aliasStr == "DipoleCurrent") {
      mMagFieldHelper.updateCurDip(static_cast<float>(o2::dcs::getValue<double>(dpcom)));
    } else if (aliasStr == "L3Polarity") {
      mMagFieldHelper.updateSignL3(o2::dcs::getValue<bool>(dpcom)); // true is negative
    } else if (aliasStr == "DipolePolarity") {
      mMagFieldHelper.updateSignDip(o2::dcs::getValue<bool>(dpcom)); // true is negative
    } else if (processEnvVar(dpcom)) {                               // environment variables
    } else if (processCollimators(dpcom)) {
    } else {
      processLHCIFDPs(dpcom);
    }
  }
  return 0;
}

//______________________________________________________________________

bool GRPDCSDPsProcessor::processCollimators(const DPCOM& dpcom)
{

  // function to process Data Points that are related to the collimators
  bool match = processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_downstream", mCollimators.mCollimators) ||
               processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_upstream", mCollimators.mCollimators) ||
               processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_downstream", mCollimators.mCollimators) ||
               processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_upstream", mCollimators.mCollimators) ||
               processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_downstream", mCollimators.mCollimators) ||
               processPairD(dpcom, "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_upstream", mCollimators.mCollimators);

  return match;
}

//______________________________________________________________________

bool GRPDCSDPsProcessor::processEnvVar(const DPCOM& dpcom)
{

  // function to process Data Points that are related to env variables
  bool match = processPairD(dpcom, "CavernTemperature", mEnvVars.mEnvVars) ||
               processPairD(dpcom, "CavernAtmosPressure", mEnvVars.mEnvVars) ||
               processPairD(dpcom, "SurfaceAtmosPressure", mEnvVars.mEnvVars) ||
               processPairD(dpcom, "CavernAtmosPressure2", mEnvVars.mEnvVars);

  return match;
}

//______________________________________________________________________
bool GRPDCSDPsProcessor::processPairD(const DPCOM& dpcom, const std::string& alias, std::unordered_map<std::string, std::vector<std::pair<uint64_t, double>>>& mapToUpdate)
{

  // function to process Data Points that is stored in a pair

  auto& vect = mapToUpdate[alias]; // we also create at this point the vector in the map if it did not exist yet

  auto& dpcomdata = dpcom.data;
  auto& dpid = dpcom.id;
  std::string aliasStr(dpid.get_alias());

  if (aliasStr == alias) {
    if (mVerbose) {
      LOG(info) << "It matches the requested string " << alias << ", let's check if the value needs to be updated";
    }
    updateVector(dpid, mapToUpdate[alias], aliasStr, dpcomdata.get_epoch_time(), o2::dcs::getValue<double>(dpcom));
    return true;
  }
  return false;
}

//______________________________________________________________________
bool GRPDCSDPsProcessor::processPairS(const DPCOM& dpcom, const std::string& alias, std::pair<uint64_t, std::string>& p, bool& flag)
{

  // function to process string Data Points that is stored in a pair

  auto& dpcomdata = dpcom.data;
  auto& dpid = dpcom.id;
  std::string aliasStr(dpid.get_alias());

  if (aliasStr == alias) {
    auto val = o2::dcs::getValue<std::string>(dpcom);
    if (mVerbose) {
      LOG(info) << "It matches the requested string " << alias << ", let's check if the value needs to be updated";
      LOG(info) << "old value = " << p.second << ", new value = " << val << ", call " << mCallSlice;
    }
    if (mCallSlice == 0 || (val != p.second)) {
      p.first = dpcomdata.get_epoch_time();
      p.second = val;
      flag = true;
      if (mVerbose) {
        LOG(info) << "Updating value " << alias << " with timestamp " << p.first;
      }
    }
    return true;
  }
  return false;
}

//______________________________________________________________________

bool GRPDCSDPsProcessor::compareToLatest(std::pair<uint64_t, double>& p, double val)
{

  // check if the content of the pair should be updated

  if (mVerbose) {
    LOG(info) << "old value = " << p.second << ", new value = " << val << ", absolute difference = " << std::abs(p.second - val) << " call " << mCallSlice;
  }
  if (mCallSlice == 0 || (std::abs(p.second - val) > 0.5e-7 * (std::abs(p.second) + std::abs(val)))) {
    if (mVerbose) {
      LOG(info) << "value will be updated";
    }
    return true;
  }
  if (mVerbose) {
    LOG(info) << "value will not be updated";
  }
  return false;
}

//______________________________________________________________________

bool GRPDCSDPsProcessor::processLHCIFDPs(const DPCOM& dpcom)
{

  // function to process the remaining LHCIF DPs

  auto& dpcomdata = dpcom.data;
  auto& dpid = dpcom.id;
  std::string aliasStr(dpid.get_alias());
  if (mVerbose) {
    LOG(info) << "Processing LHCIF DP " << aliasStr;
  }
  const auto& type = dpid.get_type();
  double val = -99999999;
  if (type == DPVAL_DOUBLE || type == RAW_DOUBLE) {
    val = o2::dcs::getValue<double>(dpcom);
  }

  for (int ibeam = 1; ibeam <= 2; ++ibeam) {
    if (aliasStr.find(fmt::format("{}{}", "LHC_IntensityBeam", ibeam)) != string::npos) {
      updateVector(dpid, mLHCInfo.mIntensityBeam[ibeam - 1], aliasStr, dpcomdata.get_epoch_time(), val);
      return true;
    }
  }

  if (aliasStr.find("BPTX") != string::npos) {
    if (aliasStr == "BPTX_deltaT_B1_B2") {
      updateVector(dpid, mLHCInfo.mBPTXdeltaT, aliasStr, dpcomdata.get_epoch_time(), val);
      return true;
    }
    if (aliasStr == "BPTX_deltaTRMS_B1_B2") {
      updateVector(dpid, mLHCInfo.mBPTXdeltaTRMS, aliasStr, dpcomdata.get_epoch_time(), val);
      return true;
    }
    for (int ibeam = 1; ibeam <= 2; ++ibeam) {
      if (aliasStr == fmt::format("{}{}", "BPTX_Phase_B", ibeam)) {
        updateVector(dpid, mLHCInfo.mBPTXPhase[ibeam - 1], aliasStr, dpcomdata.get_epoch_time(), val);
        return true;
      } else if (aliasStr == fmt::format("{}{}", "BPTX_PhaseRMS_B", ibeam)) {
        updateVector(dpid, mLHCInfo.mBPTXPhaseRMS[ibeam - 1], aliasStr, dpcomdata.get_epoch_time(), val);
        return true;
      } else if (aliasStr == fmt::format("{}{}", "BPTX_Phase_Shift_B", ibeam)) {
        updateVector(dpid, mLHCInfo.mBPTXPhaseShift[ibeam - 1], aliasStr, dpcomdata.get_epoch_time(), val);
        return true;
      }
    }
    return true;
  }
  if (aliasStr == "ALI_Lumi_Total_Inst") {
    updateVector(dpid, mLHCInfo.mInstLumi, aliasStr, dpcomdata.get_epoch_time(), val);
    return true;
  }

  for (int ibkg = 1; ibkg <= 3; ++ibkg) {
    if (aliasStr.find(fmt::format("{}{}", "ALI_Background", ibkg)) != string::npos) {
      updateVector(dpid, mLHCInfo.mBackground[ibkg - 1], aliasStr, dpcomdata.get_epoch_time(), val);
      return true;
    }
  }

  if (processPairS(dpcom, "ALI_Lumi_Source_Name", mLHCInfo.mLumiSource, mUpdateLHCIFInfo)) {
    return true;
  }
  if (processPairS(dpcom, "MACHINE_MODE", mLHCInfo.mMachineMode, mUpdateLHCIFInfo)) {
    return true;
  }
  if (processPairS(dpcom, "BEAM_MODE", mLHCInfo.mBeamMode, mUpdateLHCIFInfo)) {
    return true;
  }

  //  LOG(error) << "DP " << aliasStr << " not known for GRP, please check";
  return false;
}

//______________________________________________________________________

void GRPDCSDPsProcessor::updateMagFieldCCDB()
{
  // we need to update a CCDB for the B field --> let's prepare the CCDBInfo
  if (mVerbose) {
    LOG(info) << "At least one DP related to B field changed --> we will update CCDB with startTime " << mStartValidity;
  }
  if (mMagFieldHelper.isSet != (0x1 << 4) - 1) {
    LOG(alarm) << "Magnetic field was updated but not all fields were set: no FBI was seen?";
    mMagFieldHelper.updated = false;
    return;
  }
  mMagField.setL3Current(mMagFieldHelper.negL3 ? -mMagFieldHelper.curL3 : mMagFieldHelper.curL3);
  mMagField.setDipoleCurrent(mMagFieldHelper.negDip ? -mMagFieldHelper.curDip : mMagFieldHelper.curDip);
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mMagField, mccdbMagFieldInfo, "GLO/Config/GRPMagField", md, mStartValidity, mStartValidity + o2::ccdb::CcdbObjectInfo::MONTH);
}

//______________________________________________________________________

void GRPDCSDPsProcessor::updateLHCIFInfoCCDB()
{

  // we need to update a CCDB for the LHCIF DPs --> let's prepare the CCDBInfo

  if (mVerbose) {
    LOG(info) << "Entry related to LHCIF needs to be updated with startTime " << mStartValidity;
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mLHCInfo, mccdbLHCIFInfo, "GLO/Config/LHCIFDataPoints", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY); // valid for 3 days
  return;
}

//______________________________________________________________________

void GRPDCSDPsProcessor::updateEnvVarsCCDB()
{

  // we need to update a CCDB for the Env Variables DPs --> let's prepare the CCDBInfo

  if (mVerbose) {
    LOG(info) << "Entry related to Env Vars needs to be updated with startTime " << mStartValidity;
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mEnvVars, mccdbEnvVarsInfo, "GLO/Config/EnvVars", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY); // valid for 3 days
  return;
}

//______________________________________________________________________

void GRPDCSDPsProcessor::updateCollimatorsCCDB()
{

  // we need to update a CCDB for the Env Variables DPs --> let's prepare the CCDBInfo

  if (mVerbose) {
    LOG(info) << "Entry related to Collimators needs to be updated with startTime " << mStartValidity;
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mEnvVars, mccdbCollimatorsInfo, "GLO/Config/Collimators", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY); // valid for 3 days
  return;
}

//______________________________________________________________________

void GRPDCSDPsProcessor::printVectorInfo(const std::vector<std::pair<uint64_t, double>>& vect, bool afterUpdate)
{

  std::string stage = afterUpdate ? "after update" : "before update";
  if (mVerbose) {
    LOG(info) << "size " << stage << " : " << vect.size();
    for (const auto& it : vect) {
      LOG(info) << it.first << ", " << it.second;
    }
  }
}

//______________________________________________________________________

void GRPDCSDPsProcessor::updateVector(const DPID& dpid, std::vector<std::pair<uint64_t, double>>& vect, std::string alias, uint64_t timestamp, double val)
{
  printVectorInfo(vect, 0);
  bool updateFlag = false;

  if (mPids[dpid] == false) { // let's remove the first value when it is the leftover from the previous processing, since we now have a newer one
    if (mVerbose) {
      LOG(info) << "We will clear the existing vector, since it is the very first time we receive values for it and we have a dummy one, or the only value present is from the previous processing, so it is old";
    }
    vect.clear(); // won't hurt if the vector is empty as at the very beginning of the processing
    updateFlag = true;
  } else { // we are accumulating entries in the vector already
    updateFlag = compareToLatest(vect.back(), val);
  }

  // add new value if needed
  if (updateFlag) {
    vect.emplace_back(timestamp, val);
    if (mVerbose) {
      LOG(info) << "Adding value " << val << " with timestamp " << timestamp << " to vector for DP " << alias;
    }
  }
  printVectorInfo(vect, 1);
}
