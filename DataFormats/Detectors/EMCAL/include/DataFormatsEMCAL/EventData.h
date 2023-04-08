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
#ifndef ALICEO2_EMCAL_EVENTDATA_H_
#define ALICEO2_EMCAL_EVENTDATA_H_
#include <cstdint>
#include <gsl/span>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/MCLabel.h"

namespace o2
{

namespace emcal
{

/// \struct EventData
/// \brief EMCAL event information (per trigger)
/// \ingroup EMCALDataFormat
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since March 1st, 2020
///
/// Simple structure containing the lists of cells and clusters belonging to the
/// same collision (hardware trigger). Collision information is provided via the
/// interaction record. Attention: Lists (ranges) might be empty in case the
/// objects are not filled when creating the event structure.
template <class InputType>
struct EventData {
  InteractionRecord mInteractionRecord;                           ///< Interaction record for the trigger corresponding to this event
  gsl::span<const Cluster> mClusters;                             ///< EMCAL clusters
  gsl::span<const InputType> mCells;                              ///< EMCAL cells / digits
  gsl::span<const int> mCellIndices;                              ///< Cell indices in cluster
  std::vector<gsl::span<const o2::emcal::MCLabel>> mMCCellLabels; ///< span of MC labels for each cell
  uint64_t mTriggerBits;                                          ///< Trigger bits for the event

  /// \brief Reset event structure with empty interaction record and ranges
  void reset()
  {
    mInteractionRecord.clear();
    mClusters = gsl::span<const Cluster>();
    mCells = gsl::span<const InputType>();
    mCellIndices = gsl::span<const int>();
    mMCCellLabels = std::vector<gsl::span<const o2::emcal::MCLabel>>();
    mTriggerBits = 0;
  }

  ClassDefNV(EventData, 1);
};

} // namespace emcal

} // namespace o2

#endif // ALICEO2_EMCAL_EVENTDATA_H_
