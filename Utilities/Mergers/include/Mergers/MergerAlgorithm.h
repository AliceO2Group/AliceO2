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

#ifndef ALICEO2_MERGERS_H
#define ALICEO2_MERGERS_H

/// \file MergerAlgorithm.h
/// \brief Algorithms for merging objects.
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "ObjectStore.h"

class TObject;

namespace o2::mergers::algorithm
{

/// \brief A function which merges TObjects
void merge(TObject* const target, TObject* const other);
/// \brief A function which merges two vectors of TObjects
///
/// Iterates through others vector and searches for the object with the same name in targets vector.
/// If such item exists it is merged into the target object. If not than the item is pushed to the end
/// of targets vector.
void merge(VectorOfTObjectPtrs& targets, const VectorOfTObjectPtrs& others);

void deleteTCollections(TObject* obj);

} // namespace o2::mergers::algorithm

#endif // ALICEO2_MERGERS_H
