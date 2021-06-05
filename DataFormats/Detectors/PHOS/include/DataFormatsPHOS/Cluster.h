// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_CLUSTER_H_
#define ALICEO2_PHOS_CLUSTER_H_
#include <vector>
#include <Rtypes.h>

namespace o2
{
namespace phos
{
class Geometry;
/// \class Cluster
/// \brief Contains PHOS cluster parameters

struct CluElement {
  short absId = 0;
  bool isHG = false;
  int label = -1;
  float energy = 0.;
  float time = 0.;
  float localX = 0.;
  float localZ = 0.;
  float fraction = 0.;
  CluElement() = default;
  CluElement(short a, bool hg, float e, float t, float x, float z, int lab, float fr) : absId(a), isHG(hg), energy(e), time(t), localX(x), localZ(z), label(lab), fraction(fr) {}
};

class Cluster
{

 public:
  Cluster() = default;
  Cluster(const Cluster& clu) = default;

  ~Cluster() = default;

  /// \brief Comparison oparator, based on time and coordinates
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator<(const Cluster& other) const;
  /// \brief Comparison oparator, based on time and coordinates
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator>(const Cluster& other) const;

  float getEnergy() const { return mFullEnergy; }
  void setEnergy(float e) { mFullEnergy = e; }

  float getCoreEnergy() const { return mCoreEnergy; }
  void setCoreEnergy(float ec) { mCoreEnergy = ec; }

  float getDispersion() const { return mDispersion; }
  void setDispersion(float d) { mDispersion = d; }

  float getDistanceToBadChannel() const { return mDistToBadChannel; }
  void getElipsAxis(float lambdaShort, float lambdaLong) const
  {
    lambdaShort = mLambdaShort;
    lambdaLong = mLambdaLong;
  }
  void setElipsAxis(float lambdaShort, float lambdaLong)
  {
    mLambdaShort = lambdaShort;
    mLambdaLong = lambdaLong;
  }
  void getLocalPosition(float& posX, float& posZ) const
  {
    posX = mLocalPosX;
    posZ = mLocalPosZ;
  }
  void setLocalPosition(float posX, float posZ)
  {
    mLocalPosX = posX;
    mLocalPosZ = posZ;
  }
  int getMultiplicity() const { return mLastCluElement - mFirstCluElement; } // gets the number of digits making this cluster

  // 0: was no unfolging, -1: unfolding failed
  void setNExMax(char nmax = 1) { mNExMax = nmax; }
  char getNExMax() const { return mNExMax; }  // Number of maxima found in cluster in unfolding:
                                              // 0: was no unfolging, -1: unfolding failed
  char module() const { return mModule; }     // PHOS module of a current cluster
  void setModule(char mod) { mModule = mod; } // set PHOS module of a current cluster

  float getTime() const { return mTime; }
  void setTime(float t) { mTime = t; }

  char firedTrigger() const { return mFiredTrigger; }
  void setFiredTrigger(char t) { mFiredTrigger = t; }

  /// \brief Method to add digit to a cluster
  void addDigit() { mLastCluElement++; }

  uint32_t getFirstCluEl() const { return mFirstCluElement; }
  uint32_t getLastCluEl() const { return mLastCluElement; }
  void setFirstCluEl(uint32_t first) { mFirstCluElement = first; }
  void setLastCluEl(uint32_t last) { mLastCluElement = last; }

  // // Binary search implementation
  // std::vector<Digit>::const_iterator BinarySearch(const std::vector<Digit>* container, Digit& element);

 protected:
  char mModule = 0;               ///< Module number
  char mNExMax = -1;              ///< number of (Ex-)maxima before unfolding
  char mFiredTrigger = 0;         ///< matched with PHOS trigger: 0 no match, bit 1 with 2x2, bit 2 with 4x4
  uint32_t mFirstCluElement = -1; ///< index of the first contributing CluElement in a list
  uint32_t mLastCluElement = -1;  ///< index of the last contributing CluElement in a list
  float mLocalPosX = 0.;          ///< Center of gravity position in local module coordunates (phi direction)
  float mLocalPosZ = 0.;          ///< Center of gravity position in local module coordunates (z direction)
  float mFullEnergy = 0.;         ///< full energy of a shower
  float mCoreEnergy = 0.;         ///< energy in a shower core
  float mLambdaLong = 0.;         ///< shower ellipse axes
  float mLambdaShort = 0.;        ///< shower ellipse axes
  float mDispersion = 0.;         ///< shower dispersion
  float mTime = 0.;               ///< Time of the digit with maximal energy deposition
  float mDistToBadChannel = 999;  ///< Distance to nearest bad crystal

  ClassDefNV(Cluster, 4);
};
} // namespace phos
} // namespace o2

#endif
