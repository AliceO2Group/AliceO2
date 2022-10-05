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

#include "HMPIDCalibration/HMPIDDCSProcessor.h"
#include <TCanvas.h>

namespace o2::hmpid
{

// initialize map of DPIDs, function is called once in the
// HMPIDDCSDataProcessorSpec::init() function
void HMPIDDCSProcessor::init(const std::vector<DPID>& pids)
{
  for (const auto& it : pids) {
    mPids[it] = false;
  }
  arNmean.resize(43);
  arQthre.resize(42);
}

// process span of Datapoints, function is called repeatedly in the
// HMPIDDCSDataProcessorSpec::run() function
void HMPIDDCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // if there is no entries in span
  if (dps.size() == 0) {
    LOG(debug) << "Size = 0: ";
    return;
  }

  if (mVerbose) {
    LOG(debug) << "\n\n\nProcessing new DCS DP map\n-----------------";
  }

  if (!mFirstTimeSet) {
    mFirstTime = mStartValidity;
    mFirstTimeSet = true;
  }

  // itterate over span of datapoints
  for (const auto& dp : dps) {
    const auto& el = mPids.find(dp.id);

    // check if datapoint is within the defined map of DPs for HMPID
    if (el == mPids.end()) {
      LOG(info) << "DP " << dp.id << "Not found, will not be processed";
      continue;
    }
    // The aliasstring has been processed:
    mPids[dp.id] = true;

    // extract alias-string of DataPoints
    const std::string_view alias(dp.id.get_alias());
    const auto detectorId = alias.substr(0, 4);      // HMP_
    const auto transparencyId = alias.substr(0, 22); // HMP_TRANPLANT_MEASURE_

    // check if given dp is from HMPID
    if (transparencyId == TRANS_ID) {
      processTRANS(dp);
    } else if (detectorId == HMPID_ID) {
      processHMPID(dp);
    } else {
      LOG(debug) << "Unknown data point: " << alias;
    }
  } // end for
}

// if the string of the dp contains the HMPID-specifier "HMP_",
// but not the Transparency-specifier
void HMPIDDCSProcessor::processHMPID(const DPCOM& dp)
{

  const std::string alias(dp.id.get_alias());
  const auto hmpidString = alias.substr(alias.length() - 8);

  if (hmpidString == TEMP_IN_ID) {
    if (mVerbose) {
      LOG(info) << "Temperature_in DP: " << alias;
    }
    fillTempIn(dp);
  } else if (hmpidString == TEMP_OUT_ID) {
    if (mVerbose) {
      LOG(info) << "Temperature_out DP: " << alias;
    }
    fillTempOut(dp);
  } else if (hmpidString == HV_ID) {
    if (mVerbose) {
      LOG(info) << "HV DP: " << alias;
    }
    fillHV(dp);
  } else if (hmpidString == ENV_PRESS_ID) {
    if (mVerbose) {
      LOG(info) << "Environment Pressure DP: " << alias;
    }
    fillEnvPressure(dp);
  } else if (hmpidString == CH_PRESS_ID) {
    if (mVerbose) {
      LOG(info) << "Chamber Pressure DP: " << alias;
    }
    fillChPressure(dp);
  } else {
    LOG(info) << "Unknown data point: " << alias;
  }
}

// if the string of the dp contains the Transparency-specifier
// "HMP_TRANPLANT_MEASURE_"
void HMPIDDCSProcessor::processTRANS(const DPCOM& dp)
{
  const auto& dpid = dp.id;
  const std::string alias(dpid.get_alias());
  const auto transparencyString = alias.substr(alias.length() - 9);

  // exeptions have to be caught:
  // const auto stringNum = alias.substr(22,2);
  // const int num = stoi(stringNum);

  // Get index [0..29] of Transparency-measurement
  const int dig1 = aliasStringToInt(dpid, 22);
  const int dig2 = aliasStringToInt(dpid, 23);
  const int num = dig1 * 10 + dig2;

  if (dig1 == -1 || dig2 == -1) {
    LOG(warn) << "digits in string invalid" << dig1 << " " << dig2;
    return;
  }

  if (num < 0 || num > 29) {
    LOG(warn) << "num out of range " << num;
    return;
  }

  if (alias.substr(alias.length() - 10) == WAVE_LEN_ID) {
    waveLenVec[num].push_back(dp);
    if (mVerbose) {
      LOG(info) << "WAVE_LEN_ID DP: " << alias;
    }
  } else if (transparencyString == FREON_CELL_ID) {
    freonCellVec[num].push_back(dp);
    if (mVerbose) {
      LOG(info) << "FREON_CELL_ID DP: " << alias;
    }
  } else if (transparencyString == ARGON_CELL_ID) {
    if (mVerbose) {
      LOG(info) << "ARGON_CELL_ID DP: " << alias;
    }
    argonCellVec[num].push_back(dp);
  } else if (transparencyString == REF_ID) {
    if (alias.substr(alias.length() - 14) == ARGON_REF_ID) {
      if (mVerbose) {
        LOG(info) << "ARGON_REF_ID DP: " << alias;
      }
      argonRefVec[num].push_back(dp);
    } else if (alias.substr(alias.length() - 14) == FREON_REF_ID) {
      freonRefVec[num].push_back(dp);
      if (mVerbose) {
        LOG(info) << "FREON_REF_ID DP: " << alias;
      }
    } else {
      if (mVerbose) {
        LOG(info) << "Unknown data point: " << alias;
      }
    }
  } else {
    LOG(info) << "Datapoint not found: " << alias;
  }
}

// ======Fill
// DPCOM-entries===========================================================

// fill entries in environment pressure DPCOM-vector
void HMPIDDCSProcessor::fillEnvPressure(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();

  // check if datatype is as expected
  if (type == DeliveryType::DPVAL_DOUBLE) {
    dpVecEnv.push_back(dpcom);
  } else {
    LOG(warn) << "Invalid Datatype for Env Pressure";
  }
}

// fill entries in chamber-pressure DPCOM-vector
void HMPIDDCSProcessor::fillChPressure(const DPCOM& dpcom)
{

  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();

  if (type == DeliveryType::DPVAL_DOUBLE) {

    // find chamber number:
    auto chNum = aliasStringToInt(dpid, indexChPr);
    if (chNum < 7 && chNum >= 0) {
      dpVecCh[chNum].push_back(dpcom);
    } else {
      LOG(warn) << "Chamber Number out of range for Pressure : " << chNum;
    }
  } else {
    LOG(warn) << "Not correct datatype for Pressure : ";
  }
}

// HV in each chamber_section = 7*3 --> will result in Q_thre
void HMPIDDCSProcessor::fillHV(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();

  if (type == DeliveryType::DPVAL_DOUBLE) {
    const auto chNum = aliasStringToInt(dpid, indexChHv);
    const auto secNum = aliasStringToInt(dpid, indexSecHv);
    if (mVerbose) {
      LOGP(info, "HV ch:{} sec:{} val: {}", chNum, secNum, o2::dcs::getValue<double>(dpcom));
    }
    if (chNum < 7 && chNum >= 0) {
      if (secNum < 6 && secNum >= 0) {
        dpVecHV[6 * chNum + secNum].push_back(dpcom);
      } else {
        LOG(warn) << "Sector Number out of range for HV : " << secNum;
      }
    } else {
      LOG(warn) << "Chamber Number out of range for HV : " << chNum;
    }
  } else {
    LOG(warn) << "Not correct datatype for HV DP";
  }
}

// Temp in (T1)  in each chamber_radiator = 7*3
void HMPIDDCSProcessor::fillTempIn(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();

  if (type == DeliveryType::DPVAL_DOUBLE) {
    auto chNum = aliasStringToInt(dpid, indexChTemp);
    auto radNum = aliasStringToInt(dpid, indexRadTemp);

    // verify chamber- and raiator-numbers
    if (chNum < 7 && chNum >= 0) {
      if (radNum < 3 && radNum >= 0) {
        dpVecTempIn[3 * chNum + radNum].push_back(dpcom);
      } else {
        LOG(warn) << "Radiator Number out of range for TempIn :" << radNum;
      }
    } else {
      LOG(warn) << "Chamber Number out of range for TempIn DP :" << chNum;
    }
  } else {
    LOG(warn) << "Not correct datatype for TempIn DP ";
  }
}

// Temp out (T2), in each chamber_radiator = 7*3
void HMPIDDCSProcessor::fillTempOut(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();

  if (type == DeliveryType::DPVAL_DOUBLE) {
    auto chNum = aliasStringToInt(dpid, indexChTemp);
    auto radNum = aliasStringToInt(dpid, indexRadTemp);

    // verify chamber- and raiator-numbers
    if (chNum < 7 && chNum >= 0) {
      if (radNum < 3 && radNum >= 0) {
        dpVecTempOut[3 * chNum + radNum].push_back(dpcom);

      } else {
        LOG(warn) << "Radiator Number out of range for TempOut DP : " << radNum;
      }
    } else {
      LOG(warn) << "Chamber Number out of range for TempOut DP : " << chNum;
    }
  } else {
    LOG(warn) << "Not correct datatype for TempOut DP ";
  }
}

//==== Calculate mean photon energy=============================================

double HMPIDDCSProcessor::procTrans()
{
  for (int i = 0; i < 30; i++) {

    photEn = calculateWaveLength(i);

    if (photEn < o2::hmpid::Param::ePhotMin() ||
        photEn > o2::hmpid::Param::ePhotMax()) {
      LOG(warn) << "photon energy out of range" << photEn;
      continue; // if photon energy is out of range
    }

    // ===== evaluate phototube current for argon reference
    // ==============================================================
    refArgon = dpVector2Double(argonRefVec[i], "ARGONREF", i);
    if (refArgon == eMeanDefault) {
      if (mVerbose) {
        LOG(info) << "refArgon == defaultEMean()";
      }
      return defaultEMean();
    }

    //===== evaluate phototube current for argon cell
    //==================================================================
    cellArgon = dpVector2Double(argonCellVec[i], "ARGONCELL", i);
    if (cellArgon == eMeanDefault) {
      if (mVerbose) {
        LOG(info) << "cellArgon == defaultEMean()";
      }
      return defaultEMean();
    }

    //====evaluate phototube current for freon reference
    //================================================================
    refFreon = dpVector2Double(freonRefVec[i], "C6F14REFERENCE", i);
    if (refFreon == eMeanDefault) {
      if (mVerbose) {
        LOG(info) << "refFreon == defaultEMean()";
      }
      return defaultEMean();
    }

    // ==== evaluate phototube current for freon cell
    // ===================================================================
    cellFreon = dpVector2Double(freonCellVec[i], "C6F14CELL", i);
    if (cellFreon == eMeanDefault) {
      if (mVerbose) {
        LOG(info) << "cellFreon == defaultEMean()";
      }
      return defaultEMean();
    }

    // evaluate correction factor to calculate trasparency
    bool isEvalCorrOk = evalCorrFactor(refArgon, cellArgon, refFreon, cellFreon, photEn, i);

    // Returns false if dRefFreon * dRefArgon < 0
    if (!isEvalCorrOk) {
      return defaultEMean();
    }

    // Evaluate timestamps :

    auto s1 = getMinTime(waveLenVec[i]);
    auto s2 = getMinTime(argonRefVec[i]);
    auto s3 = getMinTime(argonCellVec[i]);
    auto s4 = getMinTime(freonRefVec[i]);
    auto s5 = getMinTime(freonCellVec[i]);

    auto e1 = getMaxTime(waveLenVec[i]);
    auto e2 = getMaxTime(argonRefVec[i]);
    auto e3 = getMaxTime(argonCellVec[i]);
    auto e4 = getMaxTime(freonRefVec[i]);
    auto e5 = getMaxTime(freonCellVec[i]);

    auto minTime = std::max({s1, s2, s3, s4, s5});
    auto maxTime = std::min({e1, e2, e3, e4, e5});

    if (minTime < mTimeEMean.first) {
      mTimeEMean.first = minTime;
    }
    if (maxTime > mTimeEMean.last) {
      mTimeEMean.last = maxTime;
    }

  } // end for

  if (sProb > 0) {
    eMean = sEnergProb / sProb;
  } else {
    if (mVerbose) {
      LOG(info) << " sProb < 0 ";
    }
    return defaultEMean();
  }

  if (eMean < o2::hmpid::Param::ePhotMin() ||
      eMean > o2::hmpid::Param::ePhotMax()) {
    LOG(info) << " eMean out of range " << eMean;
    return defaultEMean();
  }

  return eMean;

} // end ProcTrans

// ==== procTrans help-functions =============================

double HMPIDDCSProcessor::defaultEMean()
{
  if (mVerbose) {
    LOG(info) << Form(" Mean energy photon calculated ---> %f eV ", eMeanDefault);
  }
  return eMeanDefault;
}

//==== evaluate wavelenght
//=======================================================
double HMPIDDCSProcessor::calculateWaveLength(int i)
{
  // if there is no entries
  if (waveLenVec[i].size() == 0) {
    LOG(warn) << Form("No Data Point values for %i.waveLenght", i);
    return defaultEMean(); // will break this entry in foor loop
  }

  DPCOM dp = (waveLenVec[i])[0];

  // check if datatype is as expected
  if (dp.id.get_type() == DeliveryType::DPVAL_DOUBLE) {
    lambda = o2::dcs::getValue<double>(dp);
  } else {
    LOG(warn) << Form("Not correct datatype for HMP_TRANPLANT_MEASURE_%i_WAVELENGTH  --> Default E mean used!", i);
    return defaultEMean();
  }

  if (lambda < 150. || lambda > 230.) {
    LOG(warn) << Form("Wrong value for HMP_TRANPLANT_MEASURE_%i_WAVELENGTH --> Default E mean used!", i);
    return defaultEMean();
  }

  // find photon energy E in eV from radiation wavelength λ in nm
  nm2eV = 1239.842609;     // 1239.842609 from nm to eV
  photEn = nm2eV / lambda; // photon energy
  return photEn;
}

double HMPIDDCSProcessor::dpVector2Double(const std::vector<DPCOM>& dpVec,
                                          const char* dpString, int i)
{

  double dpVal;
  if (dpVec.size() == 0) {
    LOG(info) << Form(
      "No Data Point values for HMP_TRANPLANT_MEASURE_%s,%i  "
      "---> Default E mean used!",
      dpString, i);
    return defaultEMean();
  }

  DPCOM dp = dpVec[0];

  if (dp.id.get_type() == DeliveryType::DPVAL_DOUBLE) {
    dpVal = o2::dcs::getValue<double>(dp);
  } else {
    LOG(info) << Form(
      "Not correct datatype for HMP_TRANPLANT_MEASURE_%s,%i  "
      "-----> Default E mean used!",
      dpString, i);
    return defaultEMean();
  }
  return dpVal;
}

bool HMPIDDCSProcessor::evalCorrFactor(double dRefArgon, double dCellArgon,
                                       double dRefFreon, double dCellFreon,
                                       double dPhotEn, int i)
{
  // evaluate correction factor to calculate trasparency (Ref. NIMA 486 (2002)
  // 590-609)

  // Double_t aN1 = AliHMPIDParam::NIdxRad(photEn,tRefCR5);
  // Double_t aN2 = AliHMPIDParam::NMgF2Idx(photEn);
  // Double_t aN3 = 1;                              // Argon Idx

  // Double_t aR1               = ((aN1 - aN2)*(aN1 - aN2))/((aN1 + aN2)*(aN1 +
  // aN2)); Double_t aR2               = ((aN2 - aN3)*(aN2 - aN3))/((aN2 +
  // aN3)*(aN2 + aN3)); Double_t aT1               = (1 - aR1); Double_t aT2 =
  // (1 - aR2); Double_t aCorrFactor       = (aT1*aT1)/(aT2*aT2);

  // evaluate 15 mm of thickness C6F14 Trans

  aConvFactor = 1.0 - 0.3 / 1.8;

  if (dRefFreon * dRefArgon > 0) {
    aTransRad = TMath::Power((dCellFreon / dRefFreon) /
                               (dCellArgon / dRefArgon) * aCorrFactor[i],
                             aConvFactor);
  } else {
    if (mVerbose) {
      LOG(info) << "dRefFreon*dRefArgon<0" << dRefFreon * dRefArgon;
    }
    return false;
  }

  // evaluate 0.5 mm of thickness SiO2 Trans

  // TMath : Double_t  Exp(Double_t x)
  aTransSiO2 = TMath::Exp(-0.5 / o2::hmpid::Param::lAbsWin(dPhotEn));

  // evaluate 80 cm of thickness Gap (low density CH4) transparency
  aTransGap = TMath::Exp(-80. / o2::hmpid::Param::lAbsGap(dPhotEn));

  // evaluate CsI quantum efficiency
  // if dPhotEn < 6.07267, the value is zero
  aCsIQE = o2::hmpid::Param::qEffCSI(dPhotEn);

  // evaluate total convolution of all material optical properties
  aTotConvolution = aTransRad * aTransSiO2 * aTransGap * aCsIQE;

  sEnergProb += aTotConvolution * dPhotEn;
  double sProbPrev = sProb;
  sProb += aTotConvolution;

  return true;
}
// end Calculate mean photon energy=============================================

// ==== Functions that are called after run is finished ========================

// will return nullptr if there is no entry in Environment-pressure
// DPCOM-vector dpVecEnvPress
std::unique_ptr<TF1> HMPIDDCSProcessor::finalizeEnvPressure()
{
  std::unique_ptr<TF1> pEnv;
  if (dpVecEnv.size() != 0) {

    envPrFirstTime = getMinTime(dpVecEnv);
    envPrLastTime = getMaxTime(dpVecEnv);

    int cntEnvPressure = 0;
    std::unique_ptr<TGraph> pGrPenv;
    pGrPenv.reset(new TGraph);

    for (DPCOM dp : dpVecEnv) {
      auto dpVal = o2::dcs::getValue<double>(dp);
      auto time = dp.data.get_epoch_time(); //-envPrFirstTime
      pGrPenv->SetPoint(cntEnvPressure++, time, dpVal);
    }
    // envPrLastTime -= envPrFirstTime;
    // envPrFirstTime = 0;
    if (cntEnvPressure == 1) {
      pGrPenv->GetPoint(0, xP, yP);
      pEnv.reset(
        new TF1("Penv", Form("%f", yP), envPrFirstTime, envPrLastTime));
    } else {
      // envPrLastTime -= envPrFirstTime;
      // envPrFirstTime = 0;
      pEnv.reset(new TF1("Penv", "1000+x*[0]", envPrFirstTime, envPrLastTime));
      pGrPenv->Fit("Penv", "Q");
    }
    return pEnv;
  }
  LOG(info) << Form("No entries in environment pressure");
  return pEnv;
}

// returns nullptr if the element in array of DPCOM-vector has no entries
std::unique_ptr<TF1> HMPIDDCSProcessor::finalizeChPressure(int iCh)
{

  std::unique_ptr<TF1> pCh;
  if (dpVecCh[iCh].size() != 0) {
    cntChPressure = 0;

    std::unique_ptr<TGraph> pGrP;
    pGrP.reset(new TGraph);

    chPrFirstTime = getMinTime(dpVecCh[iCh]);
    chPrLastTime = getMaxTime(dpVecCh[iCh]);

    for (DPCOM dp : dpVecCh[iCh]) {
      auto dpVal = o2::dcs::getValue<double>(dp);
      auto time = dp.data.get_epoch_time(); //- chPrFirstTime
      pGrP->SetPoint(cntChPressure++, time, dpVal);
    }
    // chPrLastTime -= chPrFirstTime;
    // chPrFirstTime = 0;
    if (cntChPressure == 1) {
      pGrP->GetPoint(0, xP, yP);
      (pCh).reset(new TF1(Form("P%i", iCh), Form("%f", yP), chPrFirstTime,
                          chPrLastTime));
      pArrCh[iCh] = *(pCh.get());
    } else {
      (pCh).reset(new TF1(Form("P%i", iCh), "[0] + x*[1]", chPrFirstTime,
                          chPrLastTime));
      pGrP->Fit(Form("P%i", iCh), "Q");
      pArrCh[iCh] = *(pCh.get());
    }
    return pCh;
  }
  LOG(info) << Form("no entries in chPres for Pch%i", iCh);

  return pCh;
}

// process Tempout
void HMPIDDCSProcessor::finalizeTempOut(int iCh, int iRad)
{
  if (dpVecTempOut[3 * iCh + iRad].size() != 0) {
    cntTOut = 0;

    auto minTime = getMinTime(dpVecTempOut[3 * iCh + iRad]);
    auto maxTime = getMaxTime(dpVecTempOut[3 * iCh + iRad]);

    std::unique_ptr<TGraph> pGrTOut;
    pGrTOut.reset(new TGraph);

    for (DPCOM dp : dpVecTempOut[3 * iCh + iRad]) {
      auto dpVal = o2::dcs::getValue<double>(dp);
      auto time = dp.data.get_epoch_time(); // -minTime
      pGrTOut->SetPoint(cntTOut++, time, dpVal);
    }
    // maxTime -= minTime;
    // minTime = 0;
    std::unique_ptr<TF1> pTout;
    pTout.reset(
      new TF1(Form("Tout%i%i", iCh, iRad), "[0]+[1]*x", minTime, maxTime));

    if (cntTOut == 1) {
      pGrTOut->GetPoint(0, xP, yP);
      pTout->SetParameter(0, yP);
      pTout->SetParameter(1, 0);

    } else {
      pGrTOut->Fit(Form("Tout%i%i", iCh, iRad), "R");
    }
    pTout->SetTitle(Form(
      "Temp-Out Fit Chamber%i Radiator%i; Time [ms];Temp [C]", iCh, iRad));
    arNmean[6 * iCh + 2 * iRad + 1] = *(pTout.get());
  } else {
    LOG(info) << Form("No entries in temp-out for ch%irad%i", iCh, iRad);
  }
}

// process Tempin
void HMPIDDCSProcessor::finalizeTempIn(int iCh, int iRad)
{

  if (dpVecTempIn[3 * iCh + iRad].size() != 0) {
    cntTin = 0;

    auto minTime = getMinTime(dpVecTempIn[3 * iCh + iRad]);
    auto maxTime = getMaxTime(dpVecTempIn[3 * iCh + iRad]);

    std::unique_ptr<TGraph> pGrTIn;
    pGrTIn.reset(new TGraph);

    for (DPCOM dp : dpVecTempIn[3 * iCh + iRad]) {
      auto dpVal = o2::dcs::getValue<double>(dp);
      auto time = dp.data.get_epoch_time(); //-minTime
      pGrTIn->SetPoint(cntTin++, time, dpVal);
    }
    // maxTime -= minTime;
    // minTime = 0;
    std::unique_ptr<TF1> pTin;
    pTin.reset(
      new TF1(Form("Tin%i%i", iCh, iRad), "[0]+[1]*x", minTime, maxTime));

    if (cntTin == 1) {
      pGrTIn->GetPoint(0, xP, yP);
      pTin->SetParameter(0, yP);
      pTin->SetParameter(1, 0);
    } else {
      pGrTIn->Fit(Form("Tin%i%i", iCh, iRad), "Q");
    }

    pTin->SetTitle(Form("Temp-In Fit Chamber%i Radiator%i; Time [ms]; Temp [C]",
                        iCh, iRad));
    arNmean[6 * iCh + 2 * iRad] = *(pTin.get());

  } else {
    LOG(info) << Form("No entries in temp-in for ch%irad%i", iCh, iRad);
  }
}

// returns nullptr if the element in array of DPCOM-vector has no entries
std::unique_ptr<TF1> HMPIDDCSProcessor::finalizeHv(int iCh, int iSec)
{
  std::unique_ptr<TF1> pHvTF;
  if (dpVecHV[3 * iCh + iSec].size() != 0) {

    std::unique_ptr<TGraph> pGrHV;
    pGrHV.reset(new TGraph);
    cntHV = 0;

    hvFirstTime = getMinTime(dpVecHV[3 * iCh + iSec]);
    hvLastTime = getMaxTime(dpVecHV[3 * iCh + iSec]);

    for (DPCOM dp : dpVecHV[3 * iCh + iSec]) {
      auto dpVal = o2::dcs::getValue<double>(dp);
      auto time = dp.data.get_epoch_time(); //- hvFirstTime
      pGrHV->SetPoint(cntHV++, time, dpVal);
    }
    // hvLastTime -= hvFirstTime;
    // hvFirstTime = 0;
    if (cntHV == 1) {
      pGrHV->GetPoint(0, xP, yP);
      (pHvTF).reset(new TF1(Form("HV%i_%i", iCh, iSec), Form("%f", yP),
                            hvFirstTime, hvLastTime));
      pArrHv[3 * iCh + iSec] = *(pHvTF.get());
    } else {
      (pHvTF).reset(new TF1(Form("HV%i_%i", iCh, iSec), "[0]+x*[1]",
                            hvFirstTime, hvLastTime));
      pGrHV->Fit(Form("HV%i_%i", iCh, iSec), "Q");
      pArrHv[3 * iCh + iSec] = *(pHvTF.get());
    }

    return pHvTF;
  }
  LOG(info) << Form("No entries in HV for ch%isec%i", iCh, iSec);
  return pHvTF;
}

// Process all arrays of DPCOM-vectors, perform fits
// Called At the end of the Stream of DPs, in
// HMPIDDCSDataProcessorSpec::endOfStream() function
void HMPIDDCSProcessor::finalize()
{

  std::unique_ptr<TF1> pEnv = finalizeEnvPressure();
  for (int iCh = 0; iCh < 7; iCh++) {

    std::unique_ptr<TF1> pChPres = finalizeChPressure(iCh);

    // fills up entries 0..41 of arNmean
    for (int iRad = 0; iRad < 3; iRad++) {

      finalizeTempIn(iCh, iRad);  // 6*iCh + 2*iRad
      finalizeTempOut(iCh, iRad); // 6*iCh + 2*iRad + 1
    }

    // Fill entries in arQthre
    for (int iSec = 0; iSec < 6; iSec++) {

      std::unique_ptr<TF1> pHV = finalizeHv(iCh, iSec);

      // only fill if envP, chamP and HV datapoints are all fetched
      LOGP(info, "Finalize ch={} sec={}, pEnv={} pChPres={} pHV={}", iCh, iSec, (void*)pEnv.get(), (void*)pChPres.get(), (void*)pHV.get());
      if (pEnv != nullptr && pChPres != nullptr && pHV != nullptr) {

        const char* hvChar = (pArrHv[3 * iCh + iSec]).GetName();
        const char* chChar = (pArrCh[iCh]).GetName();
        const char* envChar = (pEnv.get())->GetName();
        const char* fFormula =
          "3*10^(3.01e-3*%s - 4.72)+170745848*exp(-(%s+%s)*0.0162012)";
        const char* fName = Form(fFormula, hvChar, chChar, envChar);

        auto minTime = std::max({envPrFirstTime, hvFirstTime, chPrFirstTime});
        auto maxTime = std::min({envPrLastTime, hvLastTime, chPrLastTime});

        std::unique_ptr<TF1> pQthre;
        pQthre.reset(new TF1(Form("HMP_QthreC%iS%i", iCh, iSec), fName, minTime,
                             maxTime));
        pQthre->SetTitle(Form(
          "Charge-Threshold Ch%iSec%i; Time [mS]; Threshold ", iCh, iSec));

        arQthre[6 * iCh + iSec] = *(pQthre.get());

      } else {
        LOG(info) << "Missing entries for HV, envPressure or chPressure";
      }
    }
  }

  LOG(info) << "======================================== ";

  double eMean = procTrans();

  std::unique_ptr<TF1> pPhotMean;
  pPhotMean.reset(new TF1("HMP_PhotEmean", Form("%f", eMean), 0,
                          mTimeEMean.last - mTimeEMean.first));

  pPhotMean->SetTitle(Form("HMP_PhotEmean; Time [mS]; Photon Energy [eV]"));

  arNmean[42] = *(pPhotMean.get());

  // Check entries of CCDB-objects
  checkEntries(arQthre, arNmean);

  // prepare CCDB: =============================================================

  std::map<std::string, std::string> md;
  md["responsible"] = "CHANGE RESPONSIBLE";
  //    o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP
  // Refractive index (T_out, T_in, mean photon energy);
  o2::calibration::Utils::prepareCCDBobjectInfo(
    arNmean, mccdbRefInfo, "HMP/Calib/RefIndex", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY);
  LOG(info) << "mStartValidity = " << mStartValidity;
  // charge threshold;
  o2::calibration::Utils::prepareCCDBobjectInfo(
    arQthre, mccdbChargeInfo, "HMP/Calib/ChargeCut", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY);
}

uint64_t HMPIDDCSProcessor::processFlags(const uint64_t flags,
                                         const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (flags & DPVAL::KEEP_ALIVE_FLAG) {
    LOG(debug) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DPVAL::END_FLAG) {
    LOG(debug) << "END_FLAG active for DP " << pid;
  }
  return 0;
}

// extract string from DPID, convert char at specified element in string
// to int
int HMPIDDCSProcessor::aliasStringToInt(const DPID& dpid, std::size_t startIndex)
{

  const std::string inputString(dpid.get_alias());
  char stringPos = inputString[startIndex];
  int charInt = ((int)stringPos) - ((int)'0');
  if (charInt < 10 && charInt >= 0) {
    return charInt;
  } else {
    return -1;
  }
}
} // namespace o2::hmpid
