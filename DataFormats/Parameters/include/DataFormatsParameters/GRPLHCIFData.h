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

/// \file GRPLHCIFData.h
/// \brief container for the LHC InterFace data

#ifndef O2_GRP_LHCIFDATA_H
#define O2_GRP_LHCIFDATA_H

#include <Rtypes.h>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "CommonTypes/Units.h"
#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/BunchFilling.h"

namespace o2
{
namespace parameters
{

class GRPLHCIFData
{

  using beamDirection = o2::constants::lhc::BeamDirection;

 public:
  GRPLHCIFData() = default;
  ~GRPLHCIFData() = default;

  static const std::unordered_map<unsigned int, unsigned int> mZtoA;

  std::pair<long, int32_t> getBeamEnergyPerZWithTime() const { return mBeamEnergyPerZ; }
  int32_t getBeamEnergyPerZ() const { return mBeamEnergyPerZ.second; }
  long getBeamEnergyPerZTime() const { return mBeamEnergyPerZ.first; }
  void setBeamEnergyPerZWithTime(std::pair<long, int32_t> p) { mBeamEnergyPerZ = p; }
  void setBeamEnergyPerZWithTime(long t, int32_t v) { mBeamEnergyPerZ = std::make_pair(t, v); }

  std::pair<long, int32_t> getFillNumberWithTime() const { return mFillNumber; }
  int32_t getFillNumber() const { return mFillNumber.second; }
  long getFillNumberTime() const { return mFillNumber.first; }
  void setFillNumberWithTime(std::pair<long, int32_t> p) { mFillNumber = p; }
  void setFillNumberWithTime(long t, int32_t v) { mFillNumber = std::make_pair(t, v); }

  const std::pair<long, std::string>& getInjectionSchemeWithTime() const { return mInjectionScheme; }
  const std::string& getInjectionScheme() const { return mInjectionScheme.second; }
  long getInjectionSchemeTime() const { return mInjectionScheme.first; }
  void setInjectionSchemeWithTime(std::pair<long, std::string> p) { mInjectionScheme = p; }
  void setInjectionSchemeWithTime(long t, std::string v) { mInjectionScheme = std::make_pair(t, v); }

  std::pair<long, int32_t> getAtomicNumberB1WithTime() const { return mAtomicNumberB1; }
  int32_t getAtomicNumberB1() const { return mAtomicNumberB1.second; }
  long getAtomicNumberB1Time() const { return mAtomicNumberB1.first; }
  void setAtomicNumberB1WithTime(std::pair<long, int32_t> p) { mAtomicNumberB1 = p; }
  void setAtomicNumberB1WithTime(long t, int32_t v) { mAtomicNumberB1 = std::make_pair(t, v); }

  std::pair<long, int32_t> getAtomicNumberB2WithTime() const { return mAtomicNumberB2; }
  int32_t getAtomicNumberB2() const { return mAtomicNumberB2.second; }
  long getAtomicNumberB2Time() const { return mAtomicNumberB2.first; }
  void setAtomicNumberB2WithTime(std::pair<long, int32_t> p) { mAtomicNumberB2 = p; }
  void setAtomicNumberB2WithTime(long t, int32_t v) { mAtomicNumberB2 = std::make_pair(t, v); }

  std::pair<long, o2::units::AngleRad_t> getCrossingAngleWithTime() const { return mCrossingAngle; }
  o2::units::AngleRad_t getCrossingAngle() const { return mCrossingAngle.second; }
  long getCrossingAngleTime() const { return mCrossingAngle.first; }
  void setCrossingAngleWithTime(std::pair<long, o2::units::AngleRad_t> p) { mCrossingAngle = p; }
  void setCrossingAngleWithTime(long t, o2::units::AngleRad_t v) { mCrossingAngle = std::make_pair(t, v); }

  const std::pair<long, o2::BunchFilling>& getBunchFillingWithTime() const { return mBunchFilling; }
  const o2::BunchFilling& getBunchFilling() const { return mBunchFilling.second; }
  long getBunchFillingTime() const { return mBunchFilling.first; }
  void setBunchFillingWithTime(std::pair<long, o2::BunchFilling> p) { mBunchFilling = p; }
  void setBunchFillingWithTime(long t, o2::BunchFilling v) { mBunchFilling = std::make_pair(t, v); }

  /// getters/setters for given beam A and Z info, encoded as A<<16+Z
  int getBeamZ(beamDirection beam) const { return mBeamAZ[static_cast<int>(beam)] & 0xffff; }
  int getBeamA(beamDirection beam) const { return mBeamAZ[static_cast<int>(beam)] >> 16; }
  float getBeamZoverA(beamDirection beam) const;
  void setBeamAZ(int a, int z, beamDirection beam) { mBeamAZ[static_cast<int>(beam)] = (a << 16) + z; }
  void setBeamAZ(beamDirection beam);
  void setBeamAZ();
  /// getters/setters for beam energy per charge and per nucleon
  float getBeamEnergyPerNucleon(beamDirection beam) const { return mBeamEnergyPerZ.second * getBeamZoverA(beam); }
  /// calculate center of mass energy per nucleon collision
  float getSqrtS() const;
  /// helper function for BunchFilling
  void translateBucketsToBCNumbers(std::vector<int32_t>& bcNb, std::vector<int32_t>& buckets, int beam);

  void print() const;
  static GRPLHCIFData* loadFrom(const std::string& grpLHCIFFileName = "");

 private:
  std::pair<long, int32_t> mBeamEnergyPerZ{}; // beam energy per charge
  std::pair<long, int32_t> mFillNumber{};
  std::pair<long, std::string> mInjectionScheme{};
  std::pair<long, int32_t> mAtomicNumberB1{}; // clockwise
  std::pair<long, int32_t> mAtomicNumberB2{}; // anticlockwise
  std::pair<long, o2::units::AngleRad_t> mCrossingAngle{};
  int mBeamAZ[beamDirection::NBeamDirections] = {0, 0}; ///< A<<16+Z for each beam
  std::pair<long, o2::BunchFilling> mBunchFilling{};    /// To hold bunch filling information

  ClassDefNV(GRPLHCIFData, 1);
};

//______________________________________________
inline float GRPLHCIFData::getBeamZoverA(beamDirection b) const
{
  // Z/A of beam 0 or 1
  int a = getBeamA(b);
  return a ? getBeamZ(b) / static_cast<float>(a) : 0.f;
}

} // namespace parameters
} // namespace o2
#endif
