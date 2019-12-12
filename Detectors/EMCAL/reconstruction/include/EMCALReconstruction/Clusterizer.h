// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterizer.h
/// \brief Definition of the EMCAL clusterizer
#ifndef ALICEO2_EMCAL_CLUSTERIZER_H
#define ALICEO2_EMCAL_CLUSTERIZER_H

#include <array>
#include "Rtypes.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"

namespace o2
{

namespace emcal
{

// Define numbers rows/columns for topological representation of cells
constexpr unsigned int NROWS = (24 + 1) * (6 + 4); // 10x supermodule rows (6 for EMCAL, 4 for DCAL). +1 accounts for topological gap between two supermodules
constexpr unsigned int NCOLS = 48 * 2 + 1;         // 2x  supermodule columns + 1 empty space in between for DCAL (not used for EMCAL)

using ClusterIndex = Short_t;

//_________________________________________________________________________
/// \class Clusterizer
/// \brief Meta class for recursive clusterizer
///
///  Implementation of same algorithm version as in AliEMCALClusterizerv2,
///  but optimized.
///
/// \author Rudiger Haake (Yale)
//_________________________________________________________________________
class Clusterizer
{
  struct cellWithE {
    cellWithE() : energy(0.), row(0), column(0) {}
    cellWithE(float e, int r, int c) : energy(e), row(r), column(c) {}
    // std::sort will require operator< to compile.
    bool operator<(cellWithE const& rhs) const
    {
      return energy < rhs.energy;
    }
    float energy;
    int row;
    int column;
  };

 public:
  Clusterizer(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE);
  Clusterizer();
  ~Clusterizer() = default;

  void initialize(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE);
  void findClusters(const std::vector<Digit>& digitArray);
  const std::vector<Cluster>* getFoundClusters() const { return &mFoundClusters; }
  const std::vector<ClusterIndex>* getFoundClustersDigitIndices() const { return &mDigitIndices; }
  void setGeometry(Geometry* geometry) { mEMCALGeometry = geometry; }
  Geometry* getGeometry() { return mEMCALGeometry; }

 private:
  void getClusterFromNeighbours(std::vector<Digit*>& clusterDigits, int row, int column);
  void getTopologicalRowColumn(const Digit& digit, int& row, int& column);
  Geometry* mEMCALGeometry = nullptr;                     //!<! pointer to geometry for utilities
  std::array<cellWithE, NROWS * NCOLS> mSeedList;         //!<! seed array
  std::array<std::array<Digit*, NCOLS>, NROWS> mDigitMap; //!<! topology arrays
  std::array<std::array<bool, NCOLS>, NROWS> mCellMask;   //!<! topology arrays

  std::vector<Cluster> mFoundClusters;     ///<  vector of cluster objects
  std::vector<ClusterIndex> mDigitIndices; ///<  vector of associated digit tower ID, ordered by cluster

  double mTimeCut;             ///<  maximum time difference between the digits inside EMC cluster
  double mTimeMin;             ///<  minimum time of physical signal in a cell/digit
  double mTimeMax;             ///<  maximum time of physical signal in a cell/digit
  double mGradientCut;         ///<  minimum energy difference to distinguish local maxima in a cluster
  bool mDoEnergyGradientCut;   ///<  cut on energy gradient
  double mThresholdSeedEnergy; ///<  minimum energy to seed a EC digit in a cluster
  double mThresholdCellEnergy; ///<  minimum energy for a digit to be a member of a cluster
  ClassDefNV(Clusterizer, 1);
};

} // namespace emcal
} // namespace o2
#endif /* ALICEO2_EMCAL_CLUSTERIZER_H */
