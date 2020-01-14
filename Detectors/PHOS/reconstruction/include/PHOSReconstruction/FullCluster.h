// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_FULLCLUSTER_H_
#define ALICEO2_PHOS_FULLCLUSTER_H_

#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace phos
{
class Geometry;
/// \class FullCluster
/// \brief PHOS cluster implementation

class FullCluster : public Cluster
{

  using Label = o2::MCCompLabel;

 public:
  FullCluster() = default;
  FullCluster(short digitAbsId, float energy, float time, int label, float scale);

  ~FullCluster() = default;

  /// \brief Method to add digit to a cluster
  /// \param digit being added, energy of this digit, may be smaller than full due to everlap
  void addDigit(short digitAbsId, float energy, float time, int label, float scale);

  void evalAll(const std::vector<Digit>* digits);

  // Get index of a digit with i
  short getDigitAbsId(Int_t i) const { return mDigitsIdList.at(i); }

  const std::vector<float>* getEnergyList() const { return &mEnergyList; }
  // gets the list of energies of digits making this recpoint
  const std::vector<int>* getTimeList() const { return &mTimeList; }

  const std::vector<std::pair<int, float>>* getLabels() const { return &mLabels; }
  char getNumberOfLocalMax(int* maxAt, float* maxAtEnergy) const; //Counts local maxima and returns their positions

  void purify(float threshold); // Removes digits below threshold

 protected:
  void evalCoreEnergy();    // computes energy within radius Rcore
  void evalLocalPosition(); // computes the position in the PHOS module
  void evalDispersion();    // computes the dispersion of the shower
  void evalElipsAxis();     // computes the axis of shower ellipsoide
  void evalTime();
  // Binary search implementation
  std::vector<Digit>::const_iterator BinarySearch(const std::vector<Digit>* container, Digit& element);

 private:
  std::vector<short> mDigitsIdList;             //!  Transient Array of digits absID
  std::vector<float> mEnergyList;               //!  Transient Array of digits energy
  std::vector<int> mTimeList;                   //!  Transient Array of digits times
  std::vector<std::pair<int, float>> mLabels;   //!  Transient Array of label indexes
  Geometry* mPHOSGeom;                          //!

  ClassDefNV(FullCluster, 1);
};
} // namespace phos
} // namespace o2

#endif
