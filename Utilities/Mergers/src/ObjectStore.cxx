// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ObjectStore.cxx
/// \brief Implementation of ObjectStore for Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/ObjectStore.h"
#include "Framework/DataRefUtils.h"
#include "Mergers/MergeInterface.h"
#include "Mergers/MergerAlgorithm.h"
#include <TObject.h>

namespace o2::mergers
{

namespace object_store_helpers
{

ObjectStore extractObjectFrom(const framework::DataRef& ref)
{
  // We try first if the object inherits MergeInterface.
  // In that case it should be used, even if it is a TObject as well.
  // todo: is there a more efficient way to do that?

  try {
    return framework::DataRefUtils::as<MergeInterface>(ref);
  } catch (std::runtime_error&) {
  }

  try {
    return TObjectPtr(framework::DataRefUtils::as<TObject>(ref).release(), algorithm::deleteTCollections);
  } catch (std::runtime_error&) {
  }

  throw std::runtime_error("The received is object is neither a TObject nor inherits MergeInterface");
}

} // namespace object_store_helpers

} // namespace o2::mergers