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

#include "DataFormatsPHOS/Digit.h"

namespace o2
{
namespace phos
{
class Geometry;
/// \class Cluster
/// \brief Contains PHOS cluster parameters

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

  void setEnergy(float e) { mFullEnergy = e; }
  float getEnergy() const { return mFullEnergy; }
  float getCoreEnergy() const { return mCoreEnergy; }
  float getDispersion() const { return mDispersion; }
  float getDistanceToBadChannel() const { return mDistToBadChannel; }
  void getElipsAxis(float* lambda) const
  {
    lambda[0] = mLambdaLong;
    lambda[1] = mLambdaShort;
  }
  void getLocalPosition(float& posX, float& posZ) const
  {
    posX = mLocalPosX;
    posZ = mLocalPosZ;
  }
  int getMultiplicity() const { return mMulDigit; } // gets the number of digits making this recpoint

  // 0: was no unfolging, -1: unfolding failed
  void setNExMax(char nmax = 1) { mNExMax = nmax; }
  char getNExMax() const { return mNExMax; } // Number of maxima found in cluster in unfolding:
                                             // 0: was no unfolging, -1: unfolding failed
  char module() const { return mModule; }    // PHOS module of a current cluster

  float getTime() const { return mTime; }

  int getLabel() const { return mLabel; } //Index in MCContainer entry
  void setLabel(int l) { mLabel = l; }

 protected:
  char mMulDigit = 0;            ///< Digit nultiplicity
  char mModule = 0;              ///< Module number
  char mNExMax = -1;             ///< number of (Ex-)maxima before unfolding
  int mLabel = -1;               ///< Ref to entry in MCTruthContainer with list of labels
  float mLocalPosX = 0.;         ///< Center of gravity position in local module coordunates (phi direction)
  float mLocalPosZ = 0.;         ///< Center of gravity position in local module coordunates (z direction)
  float mFullEnergy = 0.;        ///< full energy of a shower
  float mCoreEnergy = 0.;        ///< energy in a shower core
  float mLambdaLong = 0.;        ///< shower ellipse axes
  float mLambdaShort = 0.;       ///< shower ellipse axes
  float mDispersion = 0.;        ///< shower dispersion
  float mTime = 0.;              ///< Time of the digit with maximal energy deposition
  float mDistToBadChannel = 999; ///< Distance to nearest bad crystal

  ClassDefNV(Cluster, 1);
};
} // namespace phos
} // namespace o2

#endif
