// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_V0_H
#define ALICEO2_V0_H

#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"
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
  V0(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz,
     const o2::track::TrackParCov& trPos, const o2::track::TrackParCov& trNeg,
     GIndex trPosID, GIndex trNegID);

  GIndex getProngID(int i) const { return mProngIDs[i]; }
  void setProngID(int i, GIndex gid) { mProngIDs[i] = gid; }

  const Track& getProng(int i) const { return mProngs[i]; }
  Track& getProng(int i) { return mProngs[i]; }
  void setProng(int i, const Track& t) { mProngs[i] = t; }

  float getCosPA() const { return mCosPA; }
  void setCosPA(float c) { mCosPA = c; }

  float getDCA() const { return mDCA; }
  void setDCA(float d) { mDCA = d; }

  int getVertexID() const { return mVertexID; }
  void setVertexID(int id) { mVertexID = id; }

  float getMass2() const
  {
    return calcMass2(mProngs[0].getPID(), mProngs[1].getPID());
  }

  float calcMass2(PID pidPos, PID pidNeg) const
  {
    return calcMass2(PID::getMass2(pidPos), PID::getMass2(pidNeg));
  }

  float calcMass2(float massPos2, float massNeg2) const;

 protected:
  std::array<GIndex, 2> mProngIDs; // global IDs of prongs
  std::array<Track, 2> mProngs;    // prongs kinematics at vertex
  float mCosPA = 0;                // cos of pointing angle
  float mDCA = 9990;               // distance of closest approach of prongs
  int mVertexID = -1;              // id of parent vertex

  ClassDefNV(V0, 1);
};

} // namespace dataformats
} // namespace o2
#endif
