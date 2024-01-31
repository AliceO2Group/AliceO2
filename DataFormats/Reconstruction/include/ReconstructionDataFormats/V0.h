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

#ifndef ALICEO2_V0_H
#define ALICEO2_V0_H

#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/DecayNBodyIndex.h" // RS Remove after dropping indices in O2Physics
#include <array>
#include <Math/SVector.h>

namespace o2
{
namespace dataformats
{

class V0 : public o2::track::TrackParCov
{
 public:
  using GIndex = o2::dataformats::VtxTrackIndex;
  using Track = o2::track::TrackParCov;
  using PID = o2::track::PID;

  V0() = default;

  V0(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz,
     const o2::track::TrackParCov& trPos, const o2::track::TrackParCov& trNeg, o2::track::PID pid = o2::track::PID::K0);

  const Track& getProng(int i) const { return mProngs[i]; }
  Track& getProng(int i) { return mProngs[i]; }
  void setProng(int i, const Track& t) { mProngs[i] = t; }

  float getCosPA() const { return mCosPA; }
  void setCosPA(float c) { mCosPA = c; }

  float getDCA() const { return mDCA; }
  void setDCA(float d) { mDCA = d; }

  float calcMass2() const { return calcMass2PID(mProngs[0].getPID(), mProngs[1].getPID()); }
  float calcMass2PID(PID pidPos, PID pidNeg) const { return calcMass2(PID::getMass2(pidPos), PID::getMass2(pidNeg)); }
  float calcMass2(float massPos2, float massNeg2) const;

  float calcMass2AsPhoton() const { return calcMass2PID(PID::Electron, PID::Electron); }
  float calcMass2AsK0() const { return calcMass2PID(PID::Pion, PID::Pion); }
  float calcMass2AsLambda() const { return calcMass2PID(PID::Proton, PID::Pion); }
  float calcMass2AsAntiLambda() const { return calcMass2PID(PID::Pion, PID::Proton); }
  float calcMass2AsHyperTriton() const { return calcMass2PID(PID::Helium3, PID::Pion); }
  float calcMass2AsAntiHyperTriton() const { return calcMass2PID(PID::Pion, PID::Helium3); }
  float calcMass2AsHyperhydrog4() const { return calcMass2PID(PID::Alpha, PID::Pion); }
  float calcMass2AsAntiHyperhydrog4() const { return calcMass2PID(PID::Pion, PID::Alpha); }

  float calcR2() const { return getX() * getX() + getY() * getY(); }

 protected:
  std::array<Track, 2> mProngs;    // prongs kinematics at vertex
  float mCosPA = 0;                // cos of pointing angle
  float mDCA = 9990;               // distance of closest approach of prongs

  ClassDefNV(V0, 2);
};

} // namespace dataformats
} // namespace o2
#endif
