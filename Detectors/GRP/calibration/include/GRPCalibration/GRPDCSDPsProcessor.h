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

  std::unordered_map<std::string, std::vector<std::pair<uint64_t, double>>> mEnvVars;

  void print()
  {
    for (const auto& el : mEnvVars) {
      std::printf("%-30s\n", el.first.c_str());
      for (const auto& it : el.second) {
        std::printf("timestamp %lu   val %.3f\n", llu2lu(it.first), it.second);
      }
    }
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
  bool verbose = false;

  void updateCurL3(float v)
  {
    if (!(isSet & 0x1) || std::abs(v - curL3) > 0.1) {
      if (verbose) {
        LOG(info) << "L3 current will be updated from " << curL3 << " to " << v;
      }
      curL3 = v;
      isSet |= 0x1;
      updated = true;
    }
  }
  void updateCurDip(float v)
  {
    if (!(isSet & 0x2) || std::abs(v - curDip) > 0.1) {
      if (verbose) {
        LOG(info) << "Dipole current will be updated from " << curDip << " to " << v;
      }
      curDip = v;
      isSet |= 0x2;
      updated = true;
    }
  }
  void updateSignL3(bool v)
  {
    if (!(isSet & 0x4) || v != negL3) {
      if (verbose) {
        LOG(info) << "L3 polarity will be updated from " << negL3 << " to " << v;
      }
      negL3 = v;
      isSet |= 0x4;
      updated = true;
    }
  }
  void updateSignDip(bool v)
  {
    if (!(isSet & 0x8) || v != negDip) {
      if (verbose) {
        LOG(info) << "Dipole polarity will be updated from " << negDip << " to " << v;
      }
      negDip = v;
      isSet |= 0x8;
      updated = true;
    }
  }
};

struct GRPCollimators {

  std::unordered_map<std::string, std::vector<std::pair<uint64_t, double>>> mCollimators;

  void print()
  {

    for (const auto& el : mCollimators) {
      std::printf("%-60s\n", el.first.c_str());
      for (const auto& it : el.second) {
        std::printf("timestamp %lu   val %.3f\n", llu2lu(it.first), it.second);
      }
    }
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

  void resetAndKeepLastVector(std::vector<std::pair<uint64_t, double>>& vect)
  {
    // always check that the size is > 0 (--> begin != end) for all vectors
    if (vect.begin() != vect.end()) {
      vect.erase(vect.begin(), vect.end() - 1);
    }
  }

  void resetAndKeepLast()
  {
    for (int i = 0; i < 2; ++i) {
      resetAndKeepLastVector(mIntensityBeam[i]);
      resetAndKeepLastVector(mBPTXPhase[i]);
      resetAndKeepLastVector(mBPTXPhaseRMS[i]);
      resetAndKeepLastVector(mBPTXPhaseShift[i]);
    }
    for (int i = 0; i < 3; ++i) {
      resetAndKeepLastVector(mBackground[i]);
    }
    resetAndKeepLastVector(mInstLumi);
    resetAndKeepLastVector(mBPTXdeltaT);
    resetAndKeepLastVector(mBPTXdeltaTRMS);
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
  bool processPairD(const DPCOM& dpcom, const std::string& alias, std::unordered_map<std::string, std::vector<std::pair<uint64_t, double>>>& mapToUpdate);
  bool processPairS(const DPCOM& dpcom, const std::string& alias, std::pair<uint64_t, std::string>& p, bool& flag);
  bool compareToLatest(std::pair<uint64_t, double>& p, double val);
  bool processLHCIFDPs(const DPCOM& dpcom);

  void resetAndKeepLastLHCIFDPs() { mLHCInfo.resetAndKeepLast(); }
  void resetAndKeepLast(std::unordered_map<std::string, std::vector<std::pair<uint64_t, double>>>& mapToReset)
  {
    // keep only the latest measurement
    for (auto& el : mapToReset) {
      el.second.erase(el.second.begin(), el.second.end() - 1);
    }
  }

  void resetPIDs()
  {
    for (auto& it : mPids) {
      it.second = false;
    }
  }

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
  GRPEnvVariables& getEnvVarsObj() { return mEnvVars; }
  const o2::ccdb::CcdbObjectInfo& getccdbEnvVarsInfo() const { return mccdbEnvVarsInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbEnvVarsInfo() { return mccdbEnvVarsInfo; }
  void updateEnvVarsCCDB();

  const GRPCollimators& getCollimatorsObj() const { return mCollimators; }
  GRPCollimators& getCollimatorsObj() { return mCollimators; }
  const o2::ccdb::CcdbObjectInfo& getccdbCollimatorsInfo() const { return mccdbCollimatorsInfo; }
  o2::ccdb::CcdbObjectInfo& getccdbCollimatorsInfo() { return mccdbCollimatorsInfo; }
  void updateCollimatorsCCDB();

  void setStartValidity(long t) { mStartValidity = t; }
  void useVerboseMode() { mVerbose = true; }

  void printVectorInfo(const std::vector<std::pair<uint64_t, double>>& vect, bool afterUpdate);
  void updateVector(const DPID& dpid, std::vector<std::pair<uint64_t, double>>& vect, std::string alias, uint64_t timestamp, double val);

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

  GRPCollimators mCollimators;
  o2::ccdb::CcdbObjectInfo mccdbCollimatorsInfo;

  GRPLHCInfo mLHCInfo;
  o2::ccdb::CcdbObjectInfo mccdbLHCIFInfo;
  bool mUpdateLHCIFInfo = false;

  ClassDefNV(GRPDCSDPsProcessor, 0);
};
} // namespace grp
} // namespace o2

#endif
