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

/// \file Clusterizer.h
/// \brief Definition of the EMCAL clusterizer
#ifndef ALICEO2_EMCAL_CLUSTERIZER_H
#define ALICEO2_EMCAL_CLUSTERIZER_H

#include <array>
#include <gsl/span>
#include "Rtypes.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"

namespace o2
{

namespace emcal
{

// Define numbers rows/columns for topological representation of cells
constexpr unsigned int NROWS = (24 + 1) * (6 + 4); // 10x supermodule rows (6 for EMCAL, 4 for DCAL). +1 accounts for topological gap between two supermodules
constexpr unsigned int NCOLS = 48 * 2 + 1;         // 2x  supermodule columns + 1 empty space in between for DCAL (not used for EMCAL)

using ClusterIndex = int;

/// \class Clusterizer
/// \brief Meta class for recursive clusterizer
/// \ingroup EMCALreconstruction
/// \author Rudiger Haake (Yale)
///
///  Implementation of same algorithm version as in AliEMCALClusterizerv2,
///  but optimized.

template <class InputType>
class Clusterizer
{
  /// \struct cellWithE
  /// \brief Wrapper structure to make cell sortable in energy
  struct cellWithE {

    /// \brief Constructor
    cellWithE() : energy(0.), row(0), column(0) {}

    /// \brief Constructor
    /// \param e Energy (in GeV)
    /// \param r Row number
    /// \param c Column number
    cellWithE(float e, int r, int c) : energy(e), row(r), column(c) {}

    /// \brief Comparison lower operator comparing cells based on energy
    ///
    /// std::sort will require operator< to compile.
    ///
    /// \param rhs Cell to compare to
    /// \return True if this cell is has a lower energy, false otherwise
    bool operator<(cellWithE const& rhs) const
    {
      return energy < rhs.energy;
    }

    float energy; ///< Energy (in GeV)
    int row;      ///< Row number
    int column;   ///< Column number
  };

  /// \struct InputwithIndex
  /// \brief Link of a cell object to a cluster index
  struct InputwithIndex {
    const InputType* mInput; ///< Input cell/digit object
    ClusterIndex mIndex;     ///< index of the cluster
  };

 public:
  /// \brief Main constructor
  /// \param timeCut Max. time difference of cells in cluster in ns
  /// \param timeMin Min. accepted cell time in ns
  /// \param timeMax Max. accepted cell time in ns
  /// \param gradientCut Min. gradient value allowed in cluster splitting
  /// \param doEnergyGradientCut Apply gradient cut
  /// \param thresholdSeedE Min. energy of seed cells in GeV
  /// \param thresholdCellE Min. energy of associated cells in GeV
  Clusterizer(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE);

  /// \brief Default constructor
  Clusterizer();

  /// \brief Destructor
  ~Clusterizer() = default;

  /// \brief Clear internal buffers of found clusters and cell indices
  void clear()
  {
    mFoundClusters.clear();
    mInputIndices.clear();
  }

  /// \brief Initialize class member vars if not done in constructor
  /// \param timeCut Max. time difference of cells in cluster in ns
  /// \param timeMin Min. accepted cell time in ns
  /// \param timeMax Max. accepted cell time in ns
  /// \param gradientCut Min. gradient value allowed in cluster splitting
  /// \param doEnergyGradientCut Apply gradient cut
  /// \param thresholdSeedE Min. energy of seed cells in GeV
  /// \param thresholdCellE Min. energy of associated cells in GeV
  void initialize(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE);

  /// \brief Find clusters based on a give input collection.
  ///
  /// Start clustering from highest energy cell.
  ///
  /// \param inputArray Input collection of cells/digits
  void findClusters(const gsl::span<InputType const>& inputArray);

  /// \brief Get list of found clusters
  /// \return List of found clusters
  const std::vector<Cluster>* getFoundClusters() const { return &mFoundClusters; }

  /// \brief Get list of found cell indices
  /// \return List of found cell indices
  const std::vector<ClusterIndex>* getFoundClustersInputIndices() const { return &mInputIndices; }

  /// \brief Set EMCAL geometry
  /// \param geometry Geometry pointer
  void setGeometry(Geometry* geometry) { mEMCALGeometry = geometry; }

  /// \brief Get pointer to geometry
  /// \return EMCAL geometry
  Geometry* getGeometry() { return mEMCALGeometry; }

 private:
  /// \brief Recursively search for neighbours (EMCAL)
  /// \param[in,out] clusterInputs Cells/digits of prototype cluster
  /// \param row Row number from neighbor search in recursion step
  /// \param column Column number for neighbor search in recursion step
  void getClusterFromNeighbours(std::vector<InputwithIndex>& clusterInputs, int row, int column);

  /// \brief Get row (phi) and column (eta) of a cell/digit, values corresponding to topology
  /// \param input Input object (cell/digit)
  /// \param[out] row Topological row
  /// \param[out] column Topological column
  void getTopologicalRowColumn(const InputType& input, int& row, int& column);

  Geometry* mEMCALGeometry = nullptr;                             //!<! pointer to geometry for utilities
  std::array<cellWithE, NROWS * NCOLS> mSeedList;                 //!<! seed array
  std::array<std::array<InputwithIndex, NCOLS>, NROWS> mInputMap; //!<! topology arrays
  std::array<std::array<bool, NCOLS>, NROWS> mCellMask;           //!<! topology arrays

  std::vector<Cluster> mFoundClusters;     ///<  vector of cluster objects
  std::vector<ClusterIndex> mInputIndices; ///<  vector of associated cell/digit tower ID, ordered by cluster

  double mTimeCut;             ///<  maximum time difference between the cells/digits inside EMC cluster
  double mTimeMin;             ///<  minimum time of physical signal in a cell/digit
  double mTimeMax;             ///<  maximum time of physical signal in a cell/digit
  double mGradientCut;         ///<  minimum energy difference to distinguish local maxima in a cluster
  bool mDoEnergyGradientCut;   ///<  cut on energy gradient
  double mThresholdSeedEnergy; ///<  minimum energy to seed a EC digit/cell in a cluster
  double mThresholdCellEnergy; ///<  minimum energy for a digit/cell to be a member of a cluster
  ClassDefNV(Clusterizer, 1);
};

using ClusterizerDigits = Clusterizer<Digit>;
using ClusterizerCells = Clusterizer<Cell>;

} // namespace emcal
} // namespace o2
#endif /* ALICEO2_EMCAL_CLUSTERIZER_H */
