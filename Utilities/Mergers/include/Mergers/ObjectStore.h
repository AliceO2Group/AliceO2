// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_OBJECTSTORE_H
#define O2_OBJECTSTORE_H

/// \file ObjectStore.h
/// \brief Definition of ObjectStore for Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <variant>
#include <memory>
#include "Framework/DataRef.h"

class TObject;

namespace o2::mergers
{

class MergeInterface;

using TObjectPtr = std::shared_ptr<TObject>;
using MergeInterfacePtr = std::shared_ptr<MergeInterface>;
using ObjectStore = std::variant<std::monostate, TObjectPtr, MergeInterfacePtr>;

namespace object_store_helpers
{

/// \brief Takes a DataRef, deserializes it (if type is supported) and puts into an ObjectStore
ObjectStore extractObjectFrom(const framework::DataRef& ref);

} // namespace object_store_helpers

} // namespace o2::mergers

#endif //O2_OBJECTSTORE_H
