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
  Cluster(const Cluster& clu);

  ~Cluster() = default;

  /// \brief Comparison oparator, based on time and coordinates
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator<(const Cluster& other) const;
  /// \brief Comparison oparator, based on time and coordinates
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator>(const Cluster& other) const;

  double getEnergy() const { return mFullEnergy; }
  double getCoreEnergy() const { return mCoreEnergy; }
  double getDispersion() const { return mDispersion; }
  double getDistanceToBadChannel() const { return mDistToBadChannel; }
  void getElipsAxis(double* lambda) const
  {
    lambda[0] = mLambdaLong;
    lambda[1] = mLambdaShort;
  }
  void getLocalPosition(double& posX, double& posZ) const
  {
    posX = mLocalPosX;
    posZ = mLocalPosZ;
  }
  int getMultiplicity() const { return mMulDigit; } // gets the number of digits making this recpoint

  // 0: was no unfolging, -1: unfolding failed
  void setNExMax(short nmax = 1) { mNExMax = nmax; }
  short getNExMax() const { return mNExMax; } // Number of maxima found in cluster in unfolding:
                                              // 0: was no unfolging, -1: unfolding failed
  int getPHOSMod() const { return mModule; }  // PHOS module of a current cluster
  double getTime() const { return mTime; }

  short getLabel() const { return mLabel; }
  void setLabel(int l) { mLabel = l; }

 protected:
  int mMulDigit;            ///< Digit nultiplicity
  int mModule;              ///< Module number
  short mLabel;             ///< Ref to entry in MCTruthContainer with list of labels
  short mNExMax;            ///< number of (Ex-)maxima before unfolding
  double mLocalPosX;        ///< Center of gravity position in local module coordunates (phi direction)
  double mLocalPosZ;        ///< Center of gravity position in local module coordunates (z direction)
  double mFullEnergy;       ///< full energy of a shower
  double mCoreEnergy;       ///< energy in a shower core
  double mLambdaLong;       ///< shower ellipse axes
  double mLambdaShort;      ///< shower ellipse axes
  double mDispersion;       ///< shower dispersion
  double mTime;             ///< Time of the digit with maximal energy deposition
  double mDistToBadChannel; ///< Distance to nearest bad crystal

  ClassDefNV(Cluster, 1);
};
} // namespace phos
} // namespace o2

#endif
