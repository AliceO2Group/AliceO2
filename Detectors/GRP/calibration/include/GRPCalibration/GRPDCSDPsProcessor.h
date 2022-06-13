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

#ifndef DETECTOR_GRPDCSDPSPROCESSOR_H_
#define DETECTOR_GRPDCSDPSPROCESSOR_H_

#include <Rtypes.h>
#include <unordered_map>
#include <deque>
#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsParameters/GRPMagField.h"
#include <gsl/gsl>

/// @brief Class to process GRP DCS data points (B field, environment variables, LHCIF Data Points)

namespace o2
{
namespace grp
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

inline unsigned long llu2lu(std::uint64_t v) { return (unsigned long)v; }

struct GRPEnvVariables {

  std::pair<uint64_t, double> mCavernTemperature;
  std::pair<uint64_t, double> mCavernAtmosPressure;
  std::pair<uint64_t, double> mSurfaceAtmosPressure;
  std::pair<uint64_t, double> mCavernAtmosPressure2;

  GRPEnvVariables()
  {
    mCavernTemperature = std::make_pair(0, -999999999);
    mCavernAtmosPressure = std::make_pair(0, -999999999);
    mSurfaceAtmosPressure = std::make_pair(0, -999999999);
    mCavernAtmosPressure2 = std::make_pair(0, -999999999);
  }

  void print()
  {
    std::printf("%-30s : timestamp %lu   val %.3f\n", "Cavern Temperature", llu2lu(mCavernTemperature.first), mCavernTemperature.second);
    std::printf("%-30s : timestamp %lu   val %.3f\n", "Cavern Atm Pressure", llu2lu(mCavernAtmosPressure.first), mCavernAtmosPressure.second);
    std::printf("%-30s : timestamp %lu   val %.3f\n", "Surf Atm Pressure", llu2lu(mSurfaceAtmosPressure.first), mSurfaceAtmosPressure.second);
    std::printf("%-30s : timestamp %lu   val %.3f\n", "Cavern Atm Pressure 2", llu2lu(mCavernAtmosPressure2.first), mCavernAtmosPressure2.second);
  }

  ClassDefNV(GRPEnvVariables, 1);
};

struct MagFieldHelper {
  int isSet = 0;
  float curL3 = 0.;
  float curDip = 0.;
  bool negL3 = 0;
  bool negDip = 0;
  bool updated = false;

  void updateCurL3(float v)
  {
    if (!(isSet & 0x1) || std::abs(v - curL3) > 0.1) {
      curL3 = v;
      isSet |= 0x1;
      updated = true;
    }
  }
  void updateCurDip(float v)
  {
    if (!(isSet & 0x2) || std::abs(v - curDip) > 0.1) {
      curDip = v;
      isSet |= 0x2;
      updated = true;
    }
  }
  void updateSignL3(bool v)
  {
    if (!(isSet & 0x4) || v != negL3) {
      negL3 = v;
      isSet |= 0x4;
      updated = true;
    }
  }
  void updateSignDip(bool v)
  {
    if (!(isSet & 0x8) || v != negDip) {
      negDip = v;
      isSet |= 0x8;
      updated = true;
    }
  }
};

struct GRPCollimators {

  std::pair<uint64_t, double> mgap_downstream;
  std::pair<uint64_t, double> mgap_upstream;
  std::pair<uint64_t, double> mleft_downstream;
  std::pair<uint64_t, double> mleft_upstream;
  std::pair<uint64_t, double> mright_downstream;
  std::pair<uint64_t, double> mright_upstream;

  GRPCollimators()
  {
    mgap_downstream = std::make_pair(0, -999999999);
    mgap_upstream = std::make_pair(0, -999999999);
    mleft_downstream = std::make_pair(0, -999999999);
    mleft_upstream = std::make_pair(0, -999999999);
    mright_downstream = std::make_pair(0, -999999999);
    mright_upstream = std::make_pair(0, -999999999);
  }

  void print()
  {
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_downstream", llu2lu(mgap_downstream.first), mgap_downstream.second);
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_upstream", llu2lu(mgap_upstream.first), mgap_upstream.second);
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_downstream", llu2lu(mleft_downstream.first), mleft_downstream.second);
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_upstream", llu2lu(mleft_upstream.first), mleft_upstream.second);
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_downstream", llu2lu(mright_downstream.first), mright_downstream.second);
    std::printf("%-60s : timestamp %lu   val %.3e\n", "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_upstream", llu2lu(mright_upstream.first), mright_upstream.second);
  }

  ClassDefNV(GRPCollimators, 1);
};

struct GRPLHCInfo {

  std::array<std::vector<std::pair<uint64_t, double>>, 2> mIntensityBeam;
  std::array<std::vector<std::pair<uint64_t, double>>, 3> mBackground;
  std::vector<std::pair<uint64_t, double>> mInstLumi;
  std::vector<std::pair<uint64_t, double>> mBPTXdeltaT;
  std::vector<std::pair<uint64_t, double>> mBPTXdeltaTRMS;
  std::array<std::vector<std::pair<uint64_t, double>>, 2> mBPTXPhase;
  std::array<std::vector<std::pair<uint64_t, double>>, 2> mBPTXPhaseRMS;
  std::array<std::vector<std::pair<uint64_t, double>>, 2> mBPTXPhaseShift;
  std::pair<uint64_t, std::string> mLumiSource;  // only one value per object: when there is a change, a new object is stored
  std::pair<uint64_t, std::string> mMachineMode; // only one value per object: when there is a change, a new object is stored
  std::pair<uint64_t, std::string> mBeamMode;    // only one value per object: when there is a change, a new object is stored

  void reset()
  {
    for (int i = 0; i < 2; ++i) {
      mIntensityBeam[i].clear();
      mBPTXPhase[i].clear();
      mBPTXPhaseRMS[i].clear();
      mBPTXPhaseShift[i].clear();
    }
    for (int i = 0; i < 3; ++i) {
      mBackground[i].clear();
    }
    mInstLumi.clear();
    mBPTXdeltaT.clear();
    mBPTXdeltaTRMS.clear();
  }

  void print()
  {
    char alias[60];
    for (int i = 0; i < 2; ++i) {
      std::sprintf(alias, "LHC_IntensityBeam%d_totalIntensity", i + 1);
      std::printf("%-30s : n elements %ld\n", alias, mIntensityBeam[i].size());
      for (int iel = 0; iel < mIntensityBeam[i].size(); ++iel) {
        std::printf("timestamp %lu   val %.3e\n", llu2lu(mIntensityBeam[i].at(iel).first), mIntensityBeam[i].at(iel).second);
      }
    }
    for (int i = 0; i < 3; ++i) {
      std::sprintf(alias, "ALI_Background%d", i + 1);
      std::printf("%-30s : n elements %ld\n", alias, mBackground[i].size());
      for (int iel = 0; iel < mBackground[i].size(); ++iel) {
        std::printf("timestamp %lu   val %.3e\n", llu2lu(mBackground[i].at(iel).first), mBackground[i].at(iel).second);
      }
    }
    std::printf("%-30s : n elements %ld\n", "ALI_Lumi_Total_Inst", mInstLumi.size());
    for (int iel = 0; iel < mInstLumi.size(); ++iel) {
      std::printf("timestamp %lu   val %.3e\n", llu2lu(mInstLumi.at(iel).first), mInstLumi.at(iel).second);
    }
    std::printf("%-30s : n elements %ld\n", "BPTX_deltaT_B1_B2", mBPTXdeltaT.size());
    for (int iel = 0; iel < mBPTXdeltaT.size(); ++iel) {
      std::printf("timestamp %lu   val %.3e\n", llu2lu(mBPTXdeltaT.at(iel).first), mBPTXdeltaT.at(iel).second);
    }
    std::printf("%-30s : n elements %ld\n", "BPTX_deltaTRMS_B1_B2", mBPTXdeltaTRMS.size());
    for (int iel = 0; iel < mBPTXdeltaTRMS.size(); ++iel) {
      std::printf("timestamp %lu   val %.3e\n", llu2lu(mBPTXdeltaTRMS.at(iel).first), mBPTXdeltaTRMS.at(iel).second);
    }
    for (int i = 0; i < 2; ++i) {
      std::sprintf(alias, "BPTX_Phase_B%d", i + 1);
      std::printf("%-30s : n elements %ld\n", alias, mBPTXPhase[i].size());
      for (int iel = 0; iel < mBPTXPhase[i].size(); ++iel) {
        std::printf("timestamp %lu   val %.3e\n", llu2lu(mBPTXPhase[i].at(iel).first), mBPTXPhase[i].at(iel).second);
      }
    }
    for (int i = 0; i < 2; ++i) {
      std::sprintf(alias, "BPTX_PhaseRMS_B%d", i + 1);
      std::printf("%-30s : n elements %ld\n", alias, mBPTXPhaseRMS[i].size());
      for (int iel = 0; iel < mBPTXPhaseRMS[i].size(); ++iel) {
        std::printf("timestamp %lu   val %.3e\n", llu2lu(mBPTXPhaseRMS[i].at(iel).first), mBPTXPhaseRMS[i].at(iel).second);
      }
    }
    for (int i = 0; i < 2; ++i) {
      std::sprintf(alias, "BPTX_Phase_Shift_B%d", i + 1);
      std::printf("%-30s : n elements %ld\n", alias, mBPTXPhaseShift[i].size());
      for (int iel = 0; iel < mBPTXPhaseShift[i].size(); ++iel) {
        std::printf("timestamp %lu   val %.3e\n", llu2lu(mBPTXPhaseShift[i].at(iel).first), mBPTXPhaseShift[i].at(iel).second);
      }
    }
    std::printf("%-30s :\n", "ALI_Lumi_Source_Name");
    std::printf("timestamp %lu   val %s\n", llu2lu(mLumiSource.first), mLumiSource.second.c_str());
    std::printf("%-30s :\n", "BEAM_MODE");
    std::printf("timestamp %lu   val %s\n", llu2lu(mBeamMode.first), mBeamMode.second.c_str());
    std::printf("%-30s :\n", "MACHINE_MODE");
    std::printf("timestamp %lu   val %s\n", llu2lu(mMachineMode.first), mMachineMode.second.c_str());
  }

  ClassDefNV(GRPLHCInfo, 1);
};

class GRPDCSDPsProcessor
{
 public:
  GRPDCSDPsProcessor() = default;
  ~GRPDCSDPsProcessor() = default;

  void init(const std::vector<DPID>& pids);
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  uint64_t processFlags(uint64_t flag, const char* pid) { return 0; } // for now it is not really implemented
  bool processCollimators(const DPCOM& dpcom);
  bool processEnvVar(const DPCOM& dpcom);
  bool processPairD(const DPCOM& dpcom, const std::string& alias, std::pair<uint64_t, double>& p, bool& flag);
  bool processPairS(const DPCOM& dpcom, const std::string& alias, std::pair<uint64_t, std::string>& p, bool& flag);
  bool compareAndUpdate(std::pair<uint64_t, double>& p, const DPCOM& dpcom);
  bool processLHCIFDPs(const DPCOM& dpcom);

  void resetLHCIFDPs() { mLHCInfo.reset(); }

  const o2::parameters::GRPMagField& getMagFieldObj() const { return mMagField; }
  const o2::ccdb::CcdbObjectInfo& getccdbMagFieldInfo() const { return mccdbMagFieldInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbMagFieldInfo() { return mccdbMagFieldInfo; }
  void updateMagFieldCCDB();
  bool isMagFieldUpdated() const { return mMagFieldHelper.updated; }

  const GRPLHCInfo& getLHCIFObj() const { return mLHCInfo; }
  const o2::ccdb::CcdbObjectInfo& getccdbLHCIFInfo() const { return mccdbLHCIFInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbLHCIFInfo() { return mccdbLHCIFInfo; }
  void updateLHCIFInfoCCDB();
  bool isLHCIFInfoUpdated() const { return mUpdateLHCIFInfo; }

  const GRPEnvVariables& getEnvVarsObj() const { return mEnvVars; }
  const o2::ccdb::CcdbObjectInfo& getccdbEnvVarsInfo() const { return mccdbEnvVarsInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbEnvVarsInfo() { return mccdbEnvVarsInfo; }
  void updateEnvVarsCCDB();
  bool isEnvVarsUpdated() const { return mUpdateEnvVars; }

  const GRPCollimators& getCollimatorsObj() const { return mCollimators; }
  const o2::ccdb::CcdbObjectInfo& getccdbCollimatorsInfo() const { return mccdbCollimatorsInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbCollimatorsInfo() { return mccdbCollimatorsInfo; }
  void updateCollimatorsCCDB();
  bool isCollimatorsUpdated() const { return mUpdateCollimators; }

  void setStartValidity(long t) { mStartValidity = t; }
  void useVerboseMode() { mVerbose = true; }

 private:
  std::unordered_map<DPID, bool> mPids; // contains all PIDs for the processor, the bool
                                        // will be true if the DP was processed at least once

  long mFirstTime;         // time when a CCDB object was stored first
  long mStartValidity = 0; // TF index for processing, used to store CCDB object
  bool mFirstTimeSet = false;

  size_t mCallSlice = 0;
  bool mVerbose = false;
  MagFieldHelper mMagFieldHelper;
  o2::parameters::GRPMagField mMagField;
  o2::ccdb::CcdbObjectInfo mccdbMagFieldInfo;

  GRPEnvVariables mEnvVars;
  o2::ccdb::CcdbObjectInfo mccdbEnvVarsInfo;
  bool mUpdateEnvVars = false;

  GRPCollimators mCollimators;
  o2::ccdb::CcdbObjectInfo mccdbCollimatorsInfo;
  bool mUpdateCollimators = false;

  GRPLHCInfo mLHCInfo;
  o2::ccdb::CcdbObjectInfo mccdbLHCIFInfo;
  bool mUpdateLHCIFInfo = false;

  ClassDefNV(GRPDCSDPsProcessor, 0);
};
} // namespace grp
} // namespace o2

#endif
