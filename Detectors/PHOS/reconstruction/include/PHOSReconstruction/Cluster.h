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

#include "PHOSBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace phos
{
class Geometry;
/// \class Cluster
/// \brief PHOS cluster implementation

class Cluster
{

  using Label = o2::MCCompLabel;
  static constexpr float kSortingDelta = 1.; // used in sorting clusters
  static constexpr float kLogWeight = 4.5;   // weight used in position and disp. calculations

 public:
  Cluster() = default;
  Cluster(int digitAbsId, double energy, double time);

  ~Cluster() = default;

  /// \brief Comparison oparator, based on time and absId
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator<(const Cluster& other) const;
  /// \brief Comparison oparator, based on time and absId
  /// \param another PHOS Cluster
  /// \return result of comparison: x and z coordinates
  bool operator>(const Cluster& other) const;

  /// \brief Method to add digit to a cluster
  /// \param digit being added, energy of this digit, may be smaller than full due to everlap
  void AddDigit(int digitAbsId, double energy, double time);

  void EvalAll(const std::vector<Digit>* digits);

  // Get index of a digit with i
  int GetDigitAbsId(Int_t i) const { return mDigitsIdList.at(i); }

  double GetEnergy() const { return mFullEnergy; }
  double GetCoreEnergy() const { return mCoreEnergy; }
  double GetDispersion() const { return mDispersion; }
  double GetDistanceToBadChannel() const { return mDistToBadChannel; }
  void SetDistanceToBadChannel(double dist) { mDistToBadChannel = dist; }
  void GetElipsAxis(double* lambda) const
  {
    lambda[0] = mLambdaLong;
    lambda[1] = mLambdaShort;
  }
  const std::vector<float> GetEnergyList() const { return mEnergyList; }
  // gets the list of energies of digits making this recpoint
  const std::vector<int> GetTimeList() const { return mTimeList; }
  // gets the list of times of digits making this cluster
  void GetLocalPosition(double& posX, double& posZ) const
  {
    posX = mLocalPosX;
    posZ = mLocalPosZ;
  }
  int GetMultiplicity() const { return mMulDigit; } // gets the number of digits making this recpoint
  short GetNExMax() const { return mNExMax; }       // Number of maxima found in cluster in unfolding:
  // 0: was no unfolging, -1: unfolding failed
  void SetNExMax(short nmax = 1) { mNExMax = nmax; }
  int GetPHOSMod() const { return mModule; } // PHOS module of a current cluster
  double GetTime() const { return mTime; }

  void Purify(double threshold); // Removes digits below threshold

 protected:
  void EvalCoreEnergy(double coreRadius);
  void EvalLocalPosition(); // computes the position in the PHOS module
  void EvalDispersion();    // computes the dispersion of the shower
  void EvalElipsAxis();     // computes the axis of shower ellipsoide
  void EvalPrimaries(const std::vector<Digit>* digits);
  void EvalTime();
  // Binary search implementation
  std::vector<Digit>::const_iterator BinarySearch(const std::vector<Digit>* container, Digit& element);

 private:
  std::vector<int> mDigitsIdList;  ///< Array of digits absID
  std::vector<float> mEnergyList;  ///< Array of digits energy
  std::vector<int> mTimeList;      ///< Array of digits times
  std::vector<Label> mLabels;      ///< Array of particle labels
  std::vector<float> mLabelsEProp; ///< Array of proportios of deposited energy in total cluster E
  Geometry* mPHOSGeom;             //!

  int mMulDigit;            ///< Digit nultiplicity
  int mModule;              ///< Module number
  double mLocalPosX;        ///< Center of gravity position in local module coordunates (phi direction)
  double mLocalPosZ;        ///< Center of gravity position in local module coordunates (z direction)
  double mFullEnergy;       ///< full energy of a shower
  double mCoreEnergy;       ///< energy in a shower core
  double mLambdaLong;       ///< shower ellipse axes
  double mLambdaShort;      ///< shower ellipse axes
  double mDispersion;       ///< shower dispersion
  double mTime;             ///< Time of the digit with maximal energy deposition
  short mNExMax;            ///< number of (Ex-)maxima before unfolding
  double mDistToBadChannel; ///< Distance to nearest bad crystal
};
} // namespace phos
} // namespace o2

#endif
