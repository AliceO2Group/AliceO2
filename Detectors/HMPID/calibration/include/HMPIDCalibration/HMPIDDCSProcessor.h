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

#ifndef HMPIDDCSPROCESSOR_H
#define HMPIDDCSPROCESSOR_H

// HMPID Base
#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"

// Root classes:
#include <TF1.h>
#include <TF2.h>
#include <TGraph.h>
#include <TSystem.h>

// miscallenous libraries
#include <cmath>
#include <deque>
#include <gsl/gsl>
#include <memory>
#include <string>

// O2 includes:
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "Framework/Logger.h"

using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using DeliveryType = o2::dcs::DeliveryType;

namespace o2::hmpid
{

using namespace std::literals;
using TimeStampType = uint64_t;

class HMPIDDCSProcessor
{

 public:
  struct TimeRange {
    uint64_t first = std::numeric_limits<uint64_t>::max();
    uint64_t last = std::numeric_limits<uint64_t>::min();
  };

  HMPIDDCSProcessor() = default;
  ~HMPIDDCSProcessor() = default;

  // Process Datapoints:
  // ========================================================================================================

  // process span of DPs:
  // process DPs, fetch IDs and call processIR or processHMPID
  void init(const std::vector<DPID>& pids);

  void process(const gsl::span<const DPCOM> dps);

  void processTRANS(const DPCOM& dp);
  void processHMPID(const DPCOM& dp);

  // Fill entries of DPs==================================================
  void fillChPressure(
    const DPCOM& dpcom); // fill element[0-6] in chamber-pressure vector

  void fillEnvPressure(const DPCOM& dpcom); // fill environment-pressure vector

  // HV in each chamber_section = 7*3 --> will result in Q_thre
  void fillHV(const DPCOM& dpcom); // fill element[0-20] in HV vector

  // Temp in (T1) and out (T2), in each chamber_radiator = 7*3 :
  void fillTempIn(const DPCOM& dpcom);  // fill element[0-20] in tempIn vector
  void fillTempOut(const DPCOM& dpcom); // fill element[0-20] in tempOut vector

  // =====finalize DPs, after run is finished
  // ==================================================================================
  // functions return nullptr if there is no entry in the array of DPCOM-vectors at
  // the given element
  std::unique_ptr<TF1> finalizeEnvPressure();
  std::unique_ptr<TF1> finalizeChPressure(int iCh);
  std::unique_ptr<TF1> finalizeHv(int iCh, int iSec);
  void finalizeTempOut(int iCh, int iRad);
  void finalizeTempIn(int iCh, int iRad);

  // called from HMPIDDCSDataProcessorSpec,
  // loops over all the arrays of DPCOM-vectors, and calls the relevant
  // fill()-methods above
  void finalize();

  //===== procTrans
  //===================================================================================================
  double defaultEMean(); // just set a refractive index for C6F14 at ephot=6.675
                         // eV @ T=25 C

  double procTrans();

  bool evalCorrFactor(double dRefArgon, double dCellArgon, double dRefFreon,
                      double dCellFreon, double photEn, int i);
  double dpVector2Double(const std::vector<DPCOM>& dpVec, const char* dpString, int i);
  double calculateWaveLength(int i);

  //===== help-functions
  //================================================================================
  void setStartValidity(long t)
  {
    mStartValidity = t;
    LOGP(info, "mStartValidity {}", mStartValidity);
  }

  void resetStartValidity()
  {
    mStartValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;
  }

  // ef : runindenpendent, only used for verifying fits
  void setEndValidityRunIndependent(long t)
  {
    mEndValidity = t + 3 * o2::ccdb::CcdbObjectInfo::DAY; // ef : add some time for validity
    LOGP(info, "mEndValidity {}", mStartValidity);        // after startValidity
  }

  // ef : set end validity when Runstatus == STOP
  void setEndValidityRunSpecific(long t)
  {
    mEndValidity = t;
    LOGP(info, "mEndValidity {}", mStartValidity);
  }

  void resetEndValidity()
  {
    mEndValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;
  }

  long getStartValidity() { return mStartValidity; }

  void useVerboseMode() { mVerbose = true; }

  // convert char in aliasString to int
  int aliasStringToInt(const DPID& dpid, std::size_t startIndex);
  int subStringToInt(std::string istr, std::size_t si);
  uint64_t processFlags(const uint64_t flags, const char* pid);

  //===== DCS-CCDB methods and members Used in
  // HMPIDDCSDataProcessor===============================================================

  CcdbObjectInfo& getccdbRefInfo() { return mccdbRefInfo; }
  std::vector<TF1> getRefIndexObj() { return arNmean; }

  CcdbObjectInfo& getHmpidChargeInfo() { return mccdbChargeInfo; }
  std::vector<TF1>& getChargeCutObj() { return arQthre; }

  void clearCCDBObjects()
  {
    arQthre.clear();
    arNmean.clear();
  }

  // get methods for time-ranges
  // ===============================================================================
  // const auto& getTimeQThresh() const { return mTimeQThresh; }
  // const auto& getTimeArNmean() const { return mTimeArNmean; }

  /// / return timestamp of first fetched datapoint for a given ID (Tin/Tout,
  /// Environment pressure, HV, chamber pressure)
  TimeStampType getMinTime(const std::vector<DPCOM>& dps)
  {
    TimeStampType firstTime = std::numeric_limits<uint64_t>::max();
    for (const auto& dp : dps) {
      const auto time = dp.data.get_epoch_time();
      firstTime = std::min(firstTime, time);
    }
    return firstTime;
  }
  // return timestamp of last fetched datapoint for a given ID (Tin/Tout,
  // Environment pressure, HV, chamber pressure)
  TimeStampType getMaxTime(const std::vector<DPCOM>& dps)
  {
    TimeStampType lastTime = 0;
    for (const auto& dp : dps) {

      // check if tme of DP is greater (i.e. later) than previously latest
      // fetched DP:
      const auto time = dp.data.get_epoch_time();
      lastTime = std::max(lastTime, time);
    }
    return lastTime;
  }

  void checkEntries(const std::vector<TF1>& arQthresh,
                    const std::vector<TF1>& arrayNmean)
  {
    int cnt = 0;
    bool arQthreFull = true;

    LOG(info) << " ";
    LOG(info) << "======================================== ";
    LOG(info) << "All entries Processed";
    LOG(info) << " checking if CCDB objects are filled : ";

    for (int iCh = 0; iCh < 7; ++iCh) {
      for (int iSec = 0; iSec < 6; ++iSec) {
        auto tf = arQthresh[6 * iCh + iSec];
        const char* strCCDB = tf.GetName();
        const char* strExpected = Form("HMP_QthreC%iS%i", iCh, iSec);

        if (strcmp(strCCDB, strExpected) != 0) {
          arQthreFull = false;
          LOG(info) << "arQthre at " << 6 * iCh + iSec << "empty";
        }
        // if(tf.Getr)
      }
    }

    cnt = 0;
    bool arNmeanFull = true;
    for (int iCh = 0; iCh < 7; ++iCh) {
      for (int iRad = 0; iRad < 3; iRad += 2) {

        const char* strCcdbin = (arrayNmean[6 * iCh + 2 * iRad]).GetName();
        const char* strCcdbinOut =
          (arrayNmean[6 * iCh + 2 * iRad + 1]).GetName();

        const char* strExpectedIn = Form("Tin%i%i", iCh, iRad);
        const char* strExpectedOut = Form("Tout%i%i", iCh, iRad);

        if (strcmp(strCcdbin, strExpectedIn) != 0) {
          arNmeanFull = false;
          LOG(info) << "arNmean at " << 6 * iCh + 2 * iRad << " empty";
        }
        if (strcmp(strCcdbinOut, strExpectedOut) != 0) {
          arNmeanFull = false;
          LOG(info) << "arNmean at " << 6 * iCh + 2 * iRad + 1 << " empty";
        }
      }
    }

    if (strcmp((arrayNmean[42]).GetName(), "HMP_PhotEmean") != 0) {
      arNmeanFull = false;
    }

    if (arQthreFull) {
      LOG(info) << Form("arQthre Full Sized");
    }
    if (arNmeanFull) {
      LOG(info) << Form("arNmean Full Sized");
    }
    if (arNmeanFull && arQthreFull) {
      LOG(info) << Form("All entries of CCDB objects are filled");
    }
    LOG(info) << " ";
    LOG(info) << "======================================== ";
  }

  void clearDPsInfo()
  {
    mPids.clear();
  }

  int getRunNumberFromGRP()
  {
    return mRunNumberFromGRP;
  } // ef : just using the same as for emcal

  void setRunNumberFromGRP(int rn)
  {
    mRunNumberFromGRP = rn;
  } // ef : just using the same as for emcal
 private:
  std::unordered_map<DPID, bool> mPids;

  int mRunNumberFromGRP = -2; // ef : just using the same as for emcal

  // ======= DCS-CCDB
  // ==========================================================================================

  long mFirstTime;                                                    // time when a CCDB object was stored first
  long mStartValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; // TF index for processing, used to store CCDB object
  long mEndValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;
  long mStart = 0; // TF index for processing, used to store CCDB object
  bool mFirstTimeSet = false;

  bool mVerbose = false;

  CcdbObjectInfo mccdbRefInfo;
  CcdbObjectInfo mccdbChargeInfo;

  // objects to be stored in CCDB:
  // refractive index:
  std::vector<TF1> arNmean; // 43 21* Tin and 21*Tout (1 per radiator, 3
                            // radiators per chambers)
  // + 1 for ePhotMean (mean photon energy)
  // Charge Threshold:
  std::vector<TF1> arQthre; // 42 Qthre=f(time) one per sector

  //======= finalize() and fill() private variables
  //============================================================
  Double_t xP, yP;

  // env pressure
  int cntEnvPressure = 0;      // cnt Environment-pressure entries
  std::vector<DPCOM> dpVecEnv; // environment-pressure vector

  // ch pressure
  int cntChPressure = 0;         // cnt chamber-pressure entries in element iCh[0..6]
  std::vector<DPCOM> dpVecCh[7]; //  chamber-pressure vector [0..6]
  std::unique_ptr<TF1[]> pArrCh = std::unique_ptr<TF1[]>(new TF1[7]);

  // Temperature
  int cntTin = 0,
      cntTOut =
        0;                             // cnt tempereature entries in element i[0..20]; i = 3*iCh+iSec
  std::vector<DPCOM> dpVecTempIn[21];  //  tempIn vector [0..20]
  std::vector<DPCOM> dpVecTempOut[21]; //  tempOut vector [0..20]

  // HV
  int cntHV = 0;                  // cnt HV entries in element i[0..41];  i = iCh*6 + iSec
  std::vector<DPCOM> dpVecHV[42]; //  HV vector [0..41]; 7 chambers * 6 sectors
  std::unique_ptr<TF1[]> pArrHv = std::unique_ptr<TF1[]>(new TF1[24]);

  // procTrans variables
  // ======================================================================
  const double eMeanDefault = 6.675; // Default mean photon energy if
  // DP is invalid or not fetched

  double sEnergProb = 0, sProb = 0; // energy probaility, probability
  double eMean = 0;                 // initialize eMean (Photon energy mean) to 0

  double aCorrFactor[30] = {
    0.937575212, 0.93805688, 0.938527113, 0.938986068, 0.939433897,
    0.939870746, 0.940296755, 0.94071206, 0.941116795, 0.941511085,
    0.941895054, 0.942268821, 0.942632502, 0.942986208, 0.943330047,
    0.943664126, 0.943988544, 0.944303401, 0.944608794, 0.944904814,
    0.945191552, 0.945469097, 0.945737533, 0.945996945, 0.946247412,
    0.946489015, 0.94672183, 0.946945933, 0.947161396, 0.947368291};

  double nm2eV;  // conversion factor, nanometer to eV
  double photEn; // photon energy

  // wavelength
  double lambda;
  std::vector<DPCOM> waveLenVec[30];

  // phototube current for argon reference
  double refArgon;
  std::vector<DPCOM> argonRefVec[30];

  // phototube current for freon reference
  double refFreon;
  std::vector<DPCOM> freonRefVec[30];

  // phototube current for argon cell
  double cellArgon;
  std::vector<DPCOM> argonCellVec[30];

  // phototube current for freon cell
  double cellFreon;
  std::vector<DPCOM> freonCellVec[30];

  double aTransRad, aConvFactor; // evaluate 15 mm of thickness C6F14 Trans
  double aTransSiO2;             // evaluate 0.5 mm of thickness SiO2 Trans
  double aTransGap;              // evaluate 80 cm of thickness Gap (low density CH4)
                                 // transparency
  double aCsIQE;                 // evaluate CsI quantum efficiency
  double aTotConvolution;        // evaluate total convolution of all material optical
                                 // properties

  // indexes for getting chamber-numbers etc
  // =======================================================================================

  // Chamber Pressures
  std::size_t indexChPr = 7;

  // High Voltage
  std::size_t indexChHv = 7;
  std::size_t indexSecHv = 13;

  // Temperatures
  std::size_t indexChTemp = 7;
  std::size_t indexRadTemp = 22;

  // Timestamps and TimeRanges
  // ======================================================================================
  // timestamps of last and first  entry in vectors of DPCOMs
  uint64_t hvFirstTime, hvLastTime;
  uint64_t chPrFirstTime, chPrLastTime;
  uint64_t envPrFirstTime, envPrLastTime;

  uint64_t timeTinFirst, timeTinLast;
  uint64_t timeToutFirst, timeToutLast;

  TimeRange mTimeEMean; // Timerange for mean photon energy(procTrans)

  //======= constExpression string-literals to assign DPs to the correct method:
  //====================================================

  // check if Transparency or other HMPID specifciation
  static constexpr auto HMPID_ID{"HMP_"sv};
  static constexpr auto TRANS_ID{"HMP_TRANPLANT_MEASURE_"sv};
  // HMPID-temp, HV, pressure IDs (HMP_{"HMP_"sv};)
  static constexpr auto TEMP_OUT_ID{"OUT_TEMP"sv};
  static constexpr auto TEMP_IN_ID{"_IN_TEMP"sv};
  static constexpr auto HV_ID{"_HV_VMON"sv};
  static constexpr auto ENV_PRESS_ID{"ENV_PENV"sv};
  static constexpr auto CH_PRESS_ID{"AS_PMWPC"sv};

  // HMPID-IR IDs (TRANS_ID{"HMP_TRANPLANT_MEASURE_"sv})
  static constexpr auto WAVE_LEN_ID{"WAVELENGHT"sv};  // 0-9
  static constexpr auto REF_ID{"REFERENCE"sv};        // argonReference and freonRef
  static constexpr auto ARGON_CELL_ID{"ARGONCELL"sv}; // argon Cell reference
  static constexpr auto FREON_CELL_ID{"C6F14CELL"sv}; // fron Cell Reference

  static constexpr auto ARGON_REF_ID{"ARGONREFERENCE"sv}; // argonReference
  static constexpr auto FREON_REF_ID{"C6F14REFERENCE"sv}; // freonReference

  ClassDefNV(HMPIDDCSProcessor, 0);
}; // end class
} // namespace o2::hmpid
#endif
