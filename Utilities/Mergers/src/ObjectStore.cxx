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

/// \file ObjectStore.cxx
/// \brief Implementation of ObjectStore for Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/ObjectStore.h"
#include "Framework/DataRefUtils.h"
#include "Mergers/MergeInterface.h"
#include "Mergers/MergerAlgorithm.h"
#include "Mergers/MergerBuilder.h"
#include <TObject.h>
#include <string_view>

namespace o2::mergers
{

namespace object_store_helpers
{

constexpr static std::string_view errorPrefix = "Could not extract object to be merged: ";

template <typename... Args>
static std::string concat(Args&&... arguments)
{
  std::ostringstream ss;
  (ss << ... << arguments);
  return std::move(ss.str());
}

void* readObject(const TClass* type, o2::framework::FairTMessage& ftm)
{
  using namespace std::string_view_literals;
  auto* object = ftm.ReadObjectAny(type);

  if (object == nullptr) {
    throw std::runtime_error(concat(errorPrefix, "Failed to read object with name '"sv, type->GetName(), "' from message using ROOT serialization."sv));
  }
  return object;
}

MergeInterface* castToMergeInterface(bool inheritsFromTObject, void* object, TClass* storedClass)
{
  using namespace std::string_view_literals;
  MergeInterface* objectAsMergeInterface = inheritsFromTObject ? dynamic_cast<MergeInterface*>(static_cast<TObject*>(object)) : static_cast<MergeInterface*>(object);
  if (objectAsMergeInterface == nullptr) {
    throw std::runtime_error(concat(errorPrefix, "Could not cast '"sv, storedClass->GetName(), "' to MergeInterface"sv));
  }

  return objectAsMergeInterface;
}

std::optional<ObjectStore> extractVector(o2::framework::FairTMessage& ftm, const TClass* storedClass)
{
  if (!storedClass->InheritsFrom(TClass::GetClass(typeid(VectorOfRawTObjects)))) {
    return std::nullopt;
  }

  const auto* extractedVector = static_cast<VectorOfRawTObjects*>(readObject(storedClass, ftm));
  auto result = std::vector<TObjectPtr>{};
  result.reserve(extractedVector->size());
  std::transform(extractedVector->begin(), extractedVector->end(), std::back_inserter(result), [](const auto& rawTObject) { return TObjectPtr(rawTObject, algorithm::deleteTCollections); });
  return result;
}

ObjectStore extractObjectFrom(const framework::DataRef& ref)
{
  // We do extraction on the low level to efficiently determine if the message
  // contains an object inheriting MergeInterface or TObject. If we did it the
  // the following way and catch an exception:
  // framework::DataRefUtils::as<MergeInterface>(ref)
  // it could cause a memory leak if `ref` contained a non-owning TCollection.
  // This way we also avoid doing most of the checks twice.

  using namespace std::string_view_literals;
  using DataHeader = o2::header::DataHeader;
  if (framework::DataRefUtils::getHeader<const DataHeader*>(ref)->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
    throw std::runtime_error(concat(errorPrefix, "It is not ROOT-serialized"sv));
  }

  o2::framework::FairTMessage ftm(const_cast<char*>(ref.payload), o2::framework::DataRefUtils::getPayloadSize(ref));
  auto* storedClass = ftm.GetClass();
  if (storedClass == nullptr) {
    throw std::runtime_error(concat(errorPrefix, "Unknown stored class"sv));
  }

  if (const auto extractedVector = extractVector(ftm, storedClass)) {
    return extractedVector.value();
  }

  const bool inheritsFromMergeInterface = storedClass->InheritsFrom(TClass::GetClass(typeid(MergeInterface)));
  const bool inheritsFromTObject = storedClass->InheritsFrom(TClass::GetClass(typeid(TObject)));

  if (!inheritsFromMergeInterface && !inheritsFromTObject) {
    throw std::runtime_error(concat(errorPrefix, "Class '"sv, storedClass->GetName(), "'does not inherit from MergeInterface nor TObject"sv));
  }

  auto* object = readObject(storedClass, ftm);

  if (inheritsFromMergeInterface) {
    auto* objectAsMergeInterface = castToMergeInterface(inheritsFromTObject, object, storedClass);
    objectAsMergeInterface->postDeserialization();
    return MergeInterfacePtr(objectAsMergeInterface);
  } else {
    return TObjectPtr(static_cast<TObject*>(object), algorithm::deleteTCollections);
  }
}

VectorOfRawTObjects toRawObserverPointers(const VectorOfTObjectPtrs& vector)
{
  // NOTE: MT - it might be worth it to create custom stack allocators for this case
  VectorOfRawTObjects result{};
  result.reserve(vector.size());
  std::transform(vector.begin(), vector.end(), std::back_inserter(result), [](const auto& ptr) { return ptr.get(); });
  return result;
}

template <typename TypeToSnapshot>
struct Snapshoter {
  static bool snapshot(framework::DataAllocator& allocator, const header::DataHeader::SubSpecificationType subSpec, const ObjectStore& object)
  {
    if (!std::holds_alternative<TypeToSnapshot>(object)) {
      return false;
    }

    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerIntegralOutputBinding(), subSpec}, *std::get<TypeToSnapshot>(object));

    return true;
  }
};

template <>
struct Snapshoter<VectorOfTObjectPtrs> {
  static bool snapshot(framework::DataAllocator& allocator, const header::DataHeader::SubSpecificationType subSpec, const ObjectStore& object)
  {
    if (!std::holds_alternative<VectorOfTObjectPtrs>(object)) {
      return false;
    }

    // NOTE: it might be worth it to create custom stack allocators
    const auto& mergedVector = std::get<VectorOfTObjectPtrs>(object);
    const auto vectorToSnapshot = object_store_helpers::toRawObserverPointers(mergedVector);

    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerIntegralOutputBinding(), subSpec}, vectorToSnapshot);

    return true;
  }
};

bool snapshot(framework::DataAllocator& allocator, const header::DataHeader::SubSpecificationType subSpec, const ObjectStore& mergedObject)
{
  return Snapshoter<MergeInterfacePtr>::snapshot(allocator, subSpec, mergedObject) ||
         Snapshoter<TObjectPtr>::snapshot(allocator, subSpec, mergedObject) ||
         Snapshoter<VectorOfTObjectPtrs>::snapshot(allocator, subSpec, mergedObject);
}

} // namespace object_store_helpers

} // namespace o2::mergers
