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

/// \file CalibratorVdExB.h
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#ifndef O2_TRD_CALIBRATORCONFIGEVENTS_H
#define O2_TRD_CALIBRATORCONFIGEVENTS_H

// #include "DetectorsCalibration/TimeSlotCalibration.h"
// #include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "TRDCalibration/CalibrationParams.h"

#include "Rtypes.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"

#include <array>
#include <cstdlib>
#include <memory>

namespace o2::trd
{

class CalibratorConfigEvents
{

 public:
  CalibratorConfigEvents() { LOGP(info, "Start/end of contstructor"); }
  ~CalibratorConfigEvents() = default;

  bool hasEnoughData() const;

  void initOutput();
  void init();

  void createFile();

  void closeFile();

  // Add information from incoming partial trapconfig events that have been accumulated for 1 epn and 1 tf.
  void process(const gsl::span<MCMEvent>& trapconfigevents);

  // void retrievePrev(o2::framework::ProcessingContext& pc);
  bool timeLimitReached()
  {
    mTimeLimitCount++;
    if (mTimeLimitCount > 10) {
      mTimeLimitCount = 0;
      return true;
    } else
      return false;
  }
  const TrapConfigEvent& getCcdbObject() const { return mCCDBObject; }

  // collapse the maps holding frequency of values, into singular values for each register
  void collapseRegisterValues();

  // TrapConfigEvent& getCcdbObject() { return mCCDBObject; }

  bool isDifferent();

  void clearEventStructures();
  void setMaskedHalfChambers(const std::bitset<constants::MAXHALFCHAMBER>& hcstatusqc) { mDisabledHalfChambers = hcstatusqc; }
  std::bitset<constants::MAXHALFCHAMBER>& getDisabledHalfChambers() { return mDisabledHalfChambers; }
  int getMissingChambers() { return mDisabledHalfChambers.count() - std::count(mTimesSeenHCID.begin(), mTimesSeenHCID.end(), 0); }
  int countHCIDPresent()
  {
    int sum = 0;
    for (const auto& temp : mTimesSeenHCID) {
      LOGP(info, " {} ", temp);
      if (temp > 0)
        ++sum;
    }
    LOGP(info, "HCID SUM : {}", sum);
    return std::count_if(mTimesSeenHCID.begin(), mTimesSeenHCID.end(), [](int i) { return i > 0; });
  }

  // based off the currently listed disabled half chambers, figure out which of the currently enabled hcid we have not seen yet.
  void stillMissingHCID(std::stringstream& missinghcid);

  // sim to above but for mcm, so includes all enabled hcid, and accounts for disabled mcm.
  void stillMissingMCM(std::stringstream& missingmcm);

 private:
  bool mInitCompleted;
  uint32_t mTimeLimitCount = 0;
  uint32_t mTimeReached = 0;
  uint32_t mTimeBeforeComparison;               ///< time of accumulating data and before comparison will be done
  bool mFirstCCDBObject;                        ///< We have already saved the first config for this run.
  bool mEnableOutput{false};                    ///< enable output of configevent to a root file instead of the ccdb
  bool mSaveAllChanges = false;                 ///< Do we save all the changes to configs as they come in.
  o2::ccdb::CcdbObjectInfo mCCDBInfo;           ///< CCDB infos filled with CCDB description of accompanying CCDB calibration object
  o2::trd::TrapConfigEvent mCCDBObject;         ///< CCDB calibration  object of TrapConfigEvent
  o2::trd::TrapConfigEvent mPreviousCCDBObject; ///< CCDB calibration  object of the previously saved TrapConfigEvent

  std::bitset<constants::MAXHALFCHAMBER> mDisabledHalfChambers = 0; ///< Count of the currently enabled half chambers, used as a reference to determine the completeness of the received events.

  // both of these could be calculated in collapsing the array of array of maps, but we need this at other times as well.
  std::array<uint32_t, constants::MAXMCMCOUNT> mTimesSeenMCM;     // How many times have we seen this mcm.
  std::array<uint32_t, constants::MAXHALFCHAMBER> mTimesSeenHCID; // How many times have we seen this half chamber, count of the headers not the constituent mcm.

  // similar to above, but for hcid/mcm seen in the data stream of the rawreader (tracklets/digits)
  std::array<uint32_t, constants::MAXMCMCOUNT> mMCMSeenInData;     // How many times have we seen this mcm in the raw data stream.
  std::array<uint32_t, constants::MAXHALFCHAMBER> mHCIDSeenInData; // How many times have we seen this half chamber in the raw data stream.

  std::array<int32_t, constants::MAXMCMCOUNT> mTrapRegistersMapVectorIndex; // index of mcm in the vector of array of maps
  // TODO switch to vector of array of maps, with an array index.
  std::array<std::array<std::unordered_map<uint32_t, uint32_t>, TrapRegisters::kLastReg>, constants::MAXMCMCOUNT> mTrapRegistersFrequencyMap; // frequency map for values in the respective registers

  const TRDCalibParams& mParams{TRDCalibParams::Instance()}; ///< reference to calibration parameters

  std::unique_ptr<TFile> mOutFile{nullptr}; ///< output file
  std::unique_ptr<TTree> mOutTree{nullptr}; ///< output tree

  ClassDefNV(CalibratorConfigEvents, 1);
};

} // namespace o2::trd

#endif // O2_TRD_CALIBRATORVDEXB_H
