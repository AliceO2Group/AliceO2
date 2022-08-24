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

/// \file GRPECSObject.h
/// \brief Header of the General Run Parameters object
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_DATA_GRPECSOBJECT_H_
#define ALICEO2_DATA_GRPECSOBJECT_H_

#include <Rtypes.h>
#include <cstdint>
#include <ctime>
#include <bitset>
#include "DataFormatsParameters/ECSDataAdapters.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace parameters
{
/*
 * Collects parameters describing the run that come from ECS only.
 */

class GRPECSObject
{
  using DetID = o2::detectors::DetID;

 public:
  using timePoint = uint64_t;
  using RunType = o2::parameters::GRPECS::RunType;

  enum ROMode : int { ABSENT = 0,
                      PRESENT = 0x1,
                      CONTINUOUS = PRESENT + (0x1 << 1),
                      TRIGGERING = PRESENT + (0x1 << 2) };
  GRPECSObject() = default;
  ~GRPECSObject() = default;

  /// getters/setters for Start and Stop times according to logbook
  timePoint getTimeStart() const { return mTimeStart; }
  void setTimeStart(timePoint t) { mTimeStart = t; }

  timePoint getTimeEnd() const { return mTimeEnd; }
  void setTimeEnd(timePoint t) { mTimeEnd = t; }

  void setNHBFPerTF(uint32_t n) { mNHBFPerTF = n; }
  uint32_t getNHBFPerTF() const { return mNHBFPerTF; }

  /// getter/setter for data taking period name
  const std::string& getDataPeriod() const { return mDataPeriod; }
  void setDataPeriod(const std::string v) { mDataPeriod = v; }
  // getter/setter for run identifier
  void setRun(int r) { mRun = r; }
  int getRun() const { return mRun; }
  /// getter/setter for masks of detectors in the readout
  DetID::mask_t getDetsReadOut() const { return mDetsReadout; }
  void setDetsReadOut(DetID::mask_t mask) { mDetsReadout = mask; }
  /// getter/setter for masks of detectors with continuos readout
  DetID::mask_t getDetsContinuousReadOut() const { return mDetsContinuousRO; }
  void setDetsContinuousReadOut(DetID::mask_t mask) { mDetsContinuousRO = mask; }
  /// getter/setter for masks of detectors providing the trigger
  DetID::mask_t getDetsTrigger() const { return mDetsTrigger; }
  void setDetsTrigger(DetID::mask_t mask) { mDetsTrigger = mask; }
  /// add specific detector to the list of readout detectors
  void addDetReadOut(DetID id) { mDetsReadout |= id.getMask(); }
  /// remove specific detector from the list of readout detectors
  void remDetReadOut(DetID id)
  {
    mDetsReadout &= ~id.getMask();
    remDetContinuousReadOut(id);
    remDetTrigger(id);
  }
  /// add specific detector to the list of continuously readout detectors
  void addDetContinuousReadOut(DetID id) { mDetsContinuousRO |= id.getMask(); }
  /// remove specific detector from the list of continuouslt readout detectors
  void remDetContinuousReadOut(DetID id) { mDetsContinuousRO &= ~id.getMask(); }
  /// add specific detector to the list of triggering detectors
  void addDetTrigger(DetID id) { mDetsTrigger |= id.getMask(); }
  /// remove specific detector from the list of triggering detectors
  void remDetTrigger(DetID id) { mDetsTrigger &= ~id.getMask(); }
  /// test if detector is read out
  bool isDetReadOut(DetID id) const { return (mDetsReadout & id.getMask()) != 0; }
  /// test if detector is read out
  bool isDetContinuousReadOut(DetID id) const { return (mDetsContinuousRO & id.getMask()) != 0; }
  /// test if detector is triggering
  bool isDetTriggers(DetID id) const { return (mDetsTrigger & id.getMask()) != 0; }
  /// set detector readout mode status
  void setDetROMode(DetID id, ROMode status);
  ROMode getDetROMode(DetID id) const;

  void setRunType(RunType t) { mRunType = t; }
  auto getRunType() const { return mRunType; }

  bool isMC() const { return mIsMC; }
  void setIsMC(bool v = true) { mIsMC = v; }

  /// extra selections
  /// mask of readout detectors with addition selections. "only" overrides "skip"
  DetID::mask_t getDetsReadOut(DetID::mask_t only, DetID::mask_t skip = 0) const { return only.any() ? (mDetsReadout & only) : (mDetsReadout ^ skip); }
  /// same with comma-separate list of detector names
  DetID::mask_t getDetsReadOut(const std::string& only, const std::string& skip = "") const { return getDetsReadOut(DetID::getMask(only), DetID::getMask(skip)); }

  // methods to manipulate the list of FLPs in the run
  std::bitset<202> getListOfFLPs() const { return mFLPs; }
  void setFLPStatus(size_t flp, bool status) { mFLPs.set(flp, status); }
  bool getFLPStatus(size_t flp) const { return mFLPs.test(flp); }
  bool listOfFLPsSet() const { mFLPs.count() > 0 ? true : false; }

  /// print itself
  void print() const;

  static GRPECSObject* loadFrom(const std::string& grpecsFileName = "");
  static constexpr bool alwaysTriggeredRO(DetID::ID det) { return DefTriggeredDets[det]; }

 private:
  timePoint mTimeStart = 0; ///< DAQ_time_start entry from DAQ logbook
  timePoint mTimeEnd = 0;   ///< DAQ_time_end entry from DAQ logbook

  uint32_t mNHBFPerTF = 128; /// Number of HBFrames per TF

  DetID::mask_t mDetsReadout;       ///< mask of detectors which are read out
  DetID::mask_t mDetsContinuousRO;  ///< mask of detectors read out in continuos mode
  DetID::mask_t mDetsTrigger;       ///< mask of detectors which provide trigger input to CTP
  bool mIsMC = false;               ///< flag GRP for MC
  int mRun = 0;                     ///< run identifier
  RunType mRunType = RunType::NONE; ///< run type
  std::string mDataPeriod{};        ///< name of the period
  std::bitset<202> mFLPs{};         ///< to store which FLPs were in the processing

  // detectors which are always readout in triggered mode. Others are continuous by default but exceptionally can be triggered
  static constexpr DetID::mask_t DefTriggeredDets = DetID::getMask(DetID::TRD) | DetID::getMask(DetID::PHS) | DetID::getMask(DetID::CPV) | DetID::getMask(DetID::EMC) | DetID::getMask(DetID::HMP);

  ClassDefNV(GRPECSObject, 5);
};

} // namespace parameters
} // namespace o2

#endif
