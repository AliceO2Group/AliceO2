// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Mergers/MergeInterface.h"

class TObject;

namespace o2::mergers::algorithm
{

/// \brief A function which merges TObjects
void merge(TObject* const target, TObject* const other);
void deleteTCollections(TObject* obj);

} // namespace o2::mergers::algorithm

#endif //ALICEO2_MERGERS_H
