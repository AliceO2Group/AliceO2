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

#ifndef O2_OBJECTSTORE_H
#define O2_OBJECTSTORE_H

/// \file ObjectStore.h
/// \brief Definition of ObjectStore for Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <variant>
#include <memory>
#include <vector>
#include "Framework/DataRef.h"

class TObject;

namespace o2::mergers
{

class MergeInterface;

using TObjectPtr = std::shared_ptr<TObject>;
using VectorOfTObject = std::vector<TObject*>;
using VectorOfTObjectPtr = std::shared_ptr<VectorOfTObject>;
using MergeInterfacePtr = std::shared_ptr<MergeInterface>;
using ObjectStore = std::variant<std::monostate, TObjectPtr, VectorOfTObjectPtr, MergeInterfacePtr>;

namespace object_store_helpers
{

/// \brief Takes a DataRef, deserializes it (if type is supported) and puts into an ObjectStore
ObjectStore extractObjectFrom(const framework::DataRef& ref);

} // namespace object_store_helpers

} // namespace o2::mergers

#endif // O2_OBJECTSTORE_H
