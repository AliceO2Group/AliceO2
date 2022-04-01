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

#ifndef ALICEO2_PHOS_EVENTDATA_H_
#define ALICEO2_PHOS_EVENTDATA_H_

#include <gsl/span>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/Cluster.h"

namespace o2
{

namespace phos
{

template <class InputType>
struct EventData {
  InteractionRecord mInteractionRecord; ///< Interaction record for the trigger corresponding to this event
  gsl::span<const Cluster> mClusters;   ///< PHOS clusters
  gsl::span<const InputType> mCells;    ///< PHOS cells / digits
  gsl::span<const int> mCellIndices;    ///< Cell indices in cluster

  /// \brief Reset event structure with empty interaction record and ranges
  void reset()
  {
    mInteractionRecord.clear();
    mClusters = gsl::span<const Cluster>();
    mCells = gsl::span<const InputType>();
    mCellIndices = gsl::span<const int>();
  }

  ClassDefNV(EventData, 1);
};

} // namespace phos

} // namespace o2

#endif // ALICEO2_PHOS_EVENTDATA_H_