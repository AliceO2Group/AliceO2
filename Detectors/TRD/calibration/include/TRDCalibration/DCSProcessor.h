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

#ifndef DETECTOR_TRDDCSPROCESSOR_H_
#define DETECTOR_TRDDCSPROCESSOR_H_

#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "DataFormatsTRD/DcsCcdbObjects.h"
#include "DataFormatsTRD/Constants.h"

#include "CCDB/CcdbApi.h"
#include <Rtypes.h>
#include <unordered_map>
#include <string>
#include <bitset>
#include <gsl/gsl>

/// @brief Class to process TRD DCS data points

namespace o2
{
namespace trd
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

class DCSProcessor
{

 public:
  using TFType = uint64_t;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  DCSProcessor() = default;
  ~DCSProcessor() = default;

  // initialization based on configured DP IDs
  void init(const std::vector<DPID>& pids);

  // processing methods
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  int processFlags(uint64_t flag, const char* pid);

  // these functions prepare the CCDB objects
  bool updateGasDPsCCDB();
  bool updateVoltagesDPsCCDB();
  bool updateCurrentsDPsCCDB();
  bool updateEnvDPsCCDB();
  bool updateRunDPsCCDB();

  // signal that the CCDB object for the voltages should be updated due to change exceeding threshold
  bool shouldUpdateVoltages() const { return mShouldUpdateVoltages; }
  bool shouldUpdateRun() const { return mShouldUpdateRun; }

  // allow access to the CCDB objects from DPL processor
  CcdbObjectInfo& getccdbGasDPsInfo() { return mCcdbGasDPsInfo; }
  CcdbObjectInfo& getccdbVoltagesDPsInfo() { return mCcdbVoltagesDPsInfo; }
  CcdbObjectInfo& getccdbCurrentsDPsInfo() { return mCcdbCurrentsDPsInfo; }
  CcdbObjectInfo& getccdbEnvDPsInfo() { return mCcdbEnvDPsInfo; }
  CcdbObjectInfo& getccdbRunDPsInfo() { return mCcdbRunDPsInfo; }
  const std::unordered_map<DPID, TRDDCSMinMaxMeanInfo>& getTRDGasDPsInfo() const { return mTRDDCSGas; }
  const std::unordered_map<DPID, float>& getTRDVoltagesDPsInfo() const { return mTRDDCSVoltages; }
  const std::unordered_map<DPID, TRDDCSMinMaxMeanInfo>& getTRDCurrentsDPsInfo() const { return mTRDDCSCurrents; }
  const std::unordered_map<DPID, TRDDCSMinMaxMeanInfo>& getTRDEnvDPsInfo() const { return mTRDDCSEnv; }
  const std::unordered_map<DPID, int>& getTRDRunDPsInfo() const { return mTRDDCSRun; }

  // settings
  void setCurrentTS(TFType tf) { mCurrentTS = tf; }
  void setVerbosity(int v) { mVerbosity = v; }

  // reset methods
  void clearGasDPsInfo();
  void clearVoltagesDPsInfo();
  void clearCurrentsDPsInfo();
  void clearEnvDPsInfo();
  void clearRunDPsInfo();

  // helper functions
  int getChamberIdFromAlias(const char* alias) const;

 private:
  // the CCDB objects
  std::unordered_map<DPID, TRDDCSMinMaxMeanInfo> mTRDDCSGas;      ///< gas DPs (CO2, O2, H20 and from the chromatograph CO2, N2, Xe)
  std::unordered_map<DPID, TRDDCSMinMaxMeanInfo> mTRDDCSCurrents; ///< anode and drift currents
  std::unordered_map<DPID, float> mTRDDCSVoltages;                ///< anode and drift voltages
  std::unordered_map<DPID, TRDDCSMinMaxMeanInfo> mTRDDCSEnv;      ///< environment parameters (temperatures, pressures)
  std::unordered_map<DPID, int> mTRDDCSRun;                       ///< run number and run type
  // TODO
  // Possibly add CFG tag and chamber status here?
  // Or send errors to the InfoLogger in case CFG tag mismatches are detected for chamber which have the same FSM state?
  // For this I need more information on the chamber status - which status indicates all good and included in data taking?
  // not TODO
  // I don't think the FED ENV temperature is needed at analysis level at any point in time so I am leaving it out for now

  // helper variables
  std::unordered_map<DPID, bool> mPids;                 ///< flag for each DP whether it has been processed at least once
  std::unordered_map<DPID, uint64_t> mLastDPTimeStamps; ///< for each DP keep here the time stamp of the DP processed last
  CcdbObjectInfo mCcdbGasDPsInfo;
  CcdbObjectInfo mCcdbVoltagesDPsInfo;
  CcdbObjectInfo mCcdbCurrentsDPsInfo;
  CcdbObjectInfo mCcdbEnvDPsInfo;
  CcdbObjectInfo mCcdbRunDPsInfo;
  TFType mGasStartTS;      ///< the time stamp of the first TF which was processesd for the current GAS CCDB object
  TFType mVoltagesStartTS; ///< the time stamp of the first TF which was processesd for the current voltages CCDB object
  TFType mCurrentsStartTS; ///< the time stamp of the first TF which was processesd for the current voltages CCDB object
  TFType mEnvStartTS;
  TFType mRunStartTS;
  TFType mRunEndTS;
  TFType mCurrentTS{0}; ///< the time stamp of the TF currently being processed
  bool mGasStartTSset{false};
  bool mVoltagesStartTSSet{false};
  bool mCurrentsStartTSSet{false};
  bool mEnvStartTSSet{false};
  bool mRunStartTSSet{false};
  std::bitset<constants::MAXCHAMBER> mVoltageSet{};
  bool mShouldUpdateVoltages{false};
  bool mShouldUpdateRun{false};

  // settings
  int mVerbosity{0};

  ClassDefNV(DCSProcessor, 0);
};

} // namespace trd
} // namespace o2

#endif
