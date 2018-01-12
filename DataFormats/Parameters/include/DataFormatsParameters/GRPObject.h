// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GRPObject.h
/// \brief Header of the General Run Parameters object
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_DATA_GRPOBJECT_H_
#define ALICEO2_DATA_GRPOBJECT_H_

#include <Rtypes.h>
#include <cstdint>
#include <ctime>
#include "CommonConstants/LHCConstants.h"
#include "CommonTypes/Units.h"
#include "DetectorsBase/DetID.h"

namespace o2
{
namespace parameters
{
/*
 * Collects parameters describing the run, like the beam, magnet settings
 * masks for participating and triggered detectors etc.
 */

class GRPObject
{
  using beamDirection = o2::constants::lhc::BeamDirection;

 public:
  using timePoint = std::time_t;

  GRPObject() = default;
  ~GRPObject() = default;

  /// getters/setters for Start and Stop times according to logbook
  timePoint getTimeStart() const { return mTimeStart; }
  timePoint getTimeEnd() const { return mTimeEnd; }
  void setTimeStart(timePoint t) { mTimeStart = t; }
  void setTimeEnd(timePoint t) { mTimeEnd = t; }
  /// getters/setters for beams crossing angle (deviation from 0)
  o2::units::AngleRad_t getCrossingAngle() const { return mCrossingAngle; }
  void setCrossingAngle(o2::units::AngleRad_t v) { mCrossingAngle = v; }
  /// getters/setters for given beam A and Z info, encoded as A<<16+Z
  int getBeamZ(beamDirection beam) const { return mBeamAZ[static_cast<int>(beam)] & 0xffff; }
  int getBeamA(beamDirection beam) const { return mBeamAZ[static_cast<int>(beam)] >> 16; }
  float getBeamZ2A(beamDirection beam) const;
  void setBeamAZ(int a, int z, beamDirection beam) { mBeamAZ[static_cast<int>(beam)] = (a << 16) + z; }
  /// getters/setters for beam energy per charge and per nucleon
  void setBeamEnergyPerZ(float v) { mBeamEnergyPerZ = v; }
  float getBeamEnergyPerZ() const { return mBeamEnergyPerZ; }
  float getBeamEnergyPerNucleon(beamDirection beam) const { return mBeamEnergyPerZ * getBeamZ2A(beam); }
  /// calculate center of mass energy per nucleon collision
  float getSqrtS() const;

  /// getters/setters for magnets currents
  o2::units::Current_t getL3Current() const { return mL3Current; }
  o2::units::Current_t getDipoleCurrent() const { return mDipoleCurrent; }
  void setL3Current(o2::units::Current_t v) { mL3Current = v; }
  void setDipoleCurrent(o2::units::Current_t v) { mDipoleCurrent = v; }
  /// getter/setter for data taking period name
  const std::string& getDataPeriod() const { return mDataPeriod; }
  void setDataPeriod(const std::string v) { mDataPeriod = v; }
  /// getter/setter for LHC state in the beggining of run
  const std::string& getLHCState() const { return mLHCState; }
  void setLHCState(const std::string v) { mLHCState = v; }
  // getter/setter for run identifier
  void setRun(int r) { mRun = r; }
  int getRun() const { return mRun; }
  /// getter/setter for fill identifier
  void setFill(int f) { mFill = f; }
  int getFill() const { return mFill; }
  /// getter/setter for masks of detectors in the readout
  o2::Base::DetID::mask_t getDetsReadOut() const { return mDetsReadout; }
  void setDetsReadOut(o2::Base::DetID::mask_t mask) { mDetsReadout = mask; }
  /// getter/setter for masks of detectors providing the trigger
  o2::Base::DetID::mask_t getDetsTrigger() const { return mDetsTrigger; }
  void setDetsTrigger(o2::Base::DetID::mask_t mask) { mDetsTrigger = mask; }
  /// add specific detector to the list of readout detectors
  void addDetReadOut(o2::Base::DetID id) { mDetsReadout |= id.getMask(); }
  /// remove specific detector from the list of readout detectors
  void remDetReadOut(o2::Base::DetID id) { mDetsReadout &= ~id.getMask(); }
  /// add specific detector to the list of triggering detectors
  void addDetTrigger(o2::Base::DetID id) { mDetsTrigger |= id.getMask(); }
  /// remove specific detector from the list of triggering detectors
  void remDetTrigger(o2::Base::DetID id) { mDetsTrigger &= ~id.getMask(); }
  /// test if detector is read out
  bool isDetReadOut(o2::Base::DetID id) const { return (mDetsReadout & id.getMask()) != 0; }
  /// test if detector is triggering
  bool isDetTriggers(o2::Base::DetID id) const { return (mDetsTrigger & id.getMask()) != 0; }
  /// print itself
  void print() const;

 private:
  timePoint mTimeStart = 0; ///< DAQ_time_start entry from DAQ logbook
  timePoint mTimeEnd = 0;   ///< DAQ_time_end entry from DAQ logbook

  o2::Base::DetID::mask_t mDetsReadout; ///< mask of detectors which are read out
  o2::Base::DetID::mask_t mDetsTrigger; ///< mask of detectors which provide trigger

  o2::units::AngleRad_t mCrossingAngle = 0.f; ///< crossing angle in radians (as deviation from pi)
  o2::units::Current_t mL3Current = 0.f;      ///< signed current in L3
  o2::units::Current_t mDipoleCurrent = 0.f;  ///< signed current in Dipole
  float mBeamEnergyPerZ = 0.f;                ///< beam energy per charge (i.e. sqrt(s)/2 for pp)

  int mBeamAZ[beamDirection::NBeamDirections] = { 0, 0 }; ///< A<<16+Z for each beam

  int mRun = 0;                 ///< run identifier
  int mFill = 0;                ///< fill identifier
  std::string mDataPeriod = ""; ///< name of the period
  std::string mLHCState = "";   ///< machine state

  ClassDefNV(GRPObject, 1);
};

//______________________________________________
inline float GRPObject::getBeamZ2A(beamDirection b) const
{
  // Z/A of beam 0 or 1
  int a = getBeamA(b);
  return a ? getBeamZ(b) / static_cast<float>(a) : 0.f;
}

} // namespace parameters
} // namespace o2

#endif
