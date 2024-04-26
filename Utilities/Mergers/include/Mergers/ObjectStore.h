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
#include <Headers/DataHeader.h>

class TObject;

namespace o2
{

namespace framework
{
struct DataRef;
struct DataAllocator;
} // namespace framework

namespace mergers
{

class MergeInterface;

using TObjectPtr = std::shared_ptr<TObject>;
using VectorOfRawTObjects = std::vector<TObject*>;
using VectorOfTObjectPtrs = std::vector<TObjectPtr>;
using MergeInterfacePtr = std::shared_ptr<MergeInterface>;
using ObjectStore = std::variant<std::monostate, TObjectPtr, VectorOfTObjectPtrs, MergeInterfacePtr>;

namespace object_store_helpers
{

/// \brief Takes a DataRef, deserializes it (if type is supported) and puts into an ObjectStore
ObjectStore extractObjectFrom(const framework::DataRef& ref);

/// \brief Helper function that converts vector of smart pointers to the vector of raw pointers that is serializable.
///        Make sure that original vector lives longer than the observer vector to avoid undefined behavior.
VectorOfRawTObjects toRawObserverPointers(const VectorOfTObjectPtrs&);

/// \brief Used in FullHistorMerger's and IntegratingMerger's publish function. Checks mergedObject for every state that is NOT monostate
///        and creates snapshot of underlying object to the framework
/// \return Boolean whether the object was succesfully snapshotted or not
bool snapshot(framework::DataAllocator& allocator, const header::DataHeader::SubSpecificationType subSpec, const ObjectStore& mergedObject);

} // namespace object_store_helpers

} // namespace mergers
} // namespace o2

#endif // O2_OBJECTSTORE_H
