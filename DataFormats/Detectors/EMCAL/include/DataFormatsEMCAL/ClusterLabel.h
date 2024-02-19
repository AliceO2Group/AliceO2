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

#ifndef ALICEO2_EMCAL_CLUSTERLABEL_H_
#define ALICEO2_EMCAL_CLUSTERLABEL_H_

#include <fairlogger/Logger.h>
#include <gsl/span>
#include <vector>
#include "Rtypes.h"

namespace o2
{

namespace emcal
{

/// \class ClusterLabel
/// \brief cluster class for MC particle IDs and their respective energy fraction
/// \ingroup EMCALDataFormat
/// \author Marvin Hemmer <marvin.hemmer@cern.ch>, Goethe university Frankfurt
/// \since December 13, 2023
///

class ClusterLabel
{
 public:
  /// \struct labelWithE
  /// \brief Wrapper structure to make cluster label sortable in energy fraction
  struct labelWithE {

    /// \brief Constructor
    labelWithE() : energyFraction(0.), label(0) {}

    /// \brief Constructor
    /// \param e Energy fraction
    /// \param l MC label
    labelWithE(float e, int l) : energyFraction(e), label(l) {}

    /// \brief Comparison lower operator comparing cells based on energy
    ///
    /// std::sort will require operator>= to compile.
    ///
    /// \param rhs Label to compare to
    /// \return True if this cell is has a lower energy, false otherwise
    bool operator>=(labelWithE const& rhs) const
    {
      return energyFraction >= rhs.energyFraction;
    }

    float energyFraction; ///< Energy Fraction
    int label;            ///< MC label
  };

  // ClusterLabel() = default;
  // ~ClusterLabel() = default;
  // ClusterLabel(const ClusterLabel& clus) = default;
  // ClusterLabel& operator=(const ClusterLabel& source) = default;

  /// \brief Clear the member variables
  void clear();

  /// \brief Add label and energy fraction to the
  /// \param label MC label
  /// \param energyFraction Energy fraction
  void addValue(int label, float energyFraction);

  /// \brief Normalize the energy fraction
  /// \param factor normalization factor
  void normalize(float factor);

  /// \brief Getter for vector of labels
  std::vector<int32_t> getLabels();

  /// \brief Getter for vector of energy fractions
  std::vector<float> getEnergyFractions();

  /// \brief Sort the labels and energy fraction in descending order (largest energy fraction to smallest)
  void orderLabels();

 protected:
  std::vector<labelWithE> mClusterLabels; ///< List of MC particles that generated the cluster, paired with energy fraction
};

} // namespace emcal
} // namespace o2
#endif // ALICEO2_EMCAL_CLUSTERLABEL_H_
