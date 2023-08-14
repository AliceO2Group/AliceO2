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

#ifndef ALICEO2_3BODY_H
#define ALICEO2_3BODY_H

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"
#include <array>
#include <Math/SVector.h>

namespace o2
{
namespace dataformats
{

class Decay3Body : public o2::track::TrackParCov /// TO BE DONE: extend to generic N body vertex
{
 public:
  using Track = o2::track::TrackParCov;
  using PID = o2::track::PID;

  Decay3Body() = default;
  Decay3Body(PID pid, const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz, const Track& tr0, const Track& tr1, const Track& tr2);

  const Track& getProng(int i) const { return mProngs[i]; }
  Track& getProng(int i) { return mProngs[i]; }
  void setProng(int i, const Track& t) { mProngs[i] = t; }

  float getCosPA() const { return mCosPA; }
  void setCosPA(float c) { mCosPA = c; }

  float getDCA() const { return mDCA; }
  void setDCA(float d) { mDCA = d; }

  float calcMass2() const { return calcMass2(mProngs[0].getPID(), mProngs[1].getPID(), mProngs[2].getPID()); }
  float calcMass2(PID pid0, PID pid1, PID pid2) const { return calcMass2(pid0.getMass2(), pid1.getMass2(), pid2.getMass2()); }
  float calcMass2(float mass0, float mass1, float mass2) const;

  float calcR2() const { return getX() * getX() + getY() * getY(); }

 protected:
  std::array<Track, 3> mProngs; // prongs kinematics at vertex
  float mCosPA = 0;             // cos of pointing angle
  float mDCA = 9990;            // distance of closest approach of prongs

  ClassDefNV(Decay3Body, 1);
};

} // namespace dataformats
} // namespace o2
#endif
