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

/// \file Cluster.h
/// \brief Definition of ECal cluster class
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_CLUSTER_H
#define ALICEO2_ECAL_CLUSTER_H
#include <map>
#include <vector>
#include <Rtypes.h>
#include <TLorentzVector.h>

namespace o2
{
namespace ecal
{
class Cluster
{
 public:
  Cluster() = default;
  Cluster(const Cluster& clu) = default;
  ~Cluster() = default;

  // setters
  void addDigit(int digitIndex, int towerId, double energy);
  void setNLM(int nMax) { mNLM = nMax; }
  void setE(float energy) { mE = energy; }
  void setX(float x) { mX = x; }
  void setY(float y) { mY = y; }
  void setZ(float z) { mZ = z; }
  void setChi2(float chi2) { mChi2 = chi2; }
  void setEdgeFlag(bool isEdge) { mEdge = isEdge; }
  void addMcTrackID(int mcTrackID, float energy) { mMcTrackEnergy[mcTrackID] += energy; }

  // getters
  const std::map<int, float>& getMcTrackEnergy() { return mMcTrackEnergy; }
  int getMultiplicity() const { return mDigitIndex.size(); }
  int getDigitIndex(int i) const { return mDigitIndex[i]; }
  int getDigitTowerId(int i) const { return mDigitTowerId[i]; }
  float getDigitEnergy(int i) const { return mDigitEnergy[i]; }
  float getNLM() const { return mNLM; }
  float getTime() const { return mTime; }
  float getE() const { return mE; }
  float getX() const { return mX; }
  float getY() const { return mY; }
  float getZ() const { return mZ; }
  float getR() const { return std::sqrt(mX * mX + mY * mY); }
  float getTheta() const { return std::atan2(getR(), mZ); }
  float getEta() const { return -std::log(std::tan(getTheta() / 2.)); }
  float getPhi() const { return std::atan2(mY, mX); }
  float getChi2() const { return mChi2; }
  bool isAtTheEdge() const { return mEdge; }
  int getMcTrackID() const;
  TLorentzVector getMomentum() const;

 private:
  std::vector<int> mDigitIndex;        // vector of digit indices in digits vector
  std::vector<int> mDigitTowerId;      // vector of corresponding digit tower Ids
  std::vector<float> mDigitEnergy;     // vector of corresponding digit energies
  std::map<int, float> mMcTrackEnergy; // MC track indices and corresponding energies
  int mNLM = 0;                        // number of local maxima in the initial cluster
  float mTime = 0;                     // cluster time
  float mE = 0;                        // cluster energy
  float mX = 0;                        // estimated x-coordinate
  float mY = 0;                        // estimated y-ccordinate
  float mZ = 0;                        // estimated z-ccordinate
  float mChi2 = 0;                     // chi2 wrt EM shape
  bool mEdge = 0;                      // set to true if one of cluster digits is at the chamber edge
  ClassDefNV(Cluster, 1);
};
} // namespace ecal
} // namespace o2

#endif // ALICEO2_ECAL_CLUSTER_H
