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
#include <TObject.h>

namespace o2::mergers
{

namespace object_store_helpers
{

ObjectStore extractObjectFrom(const framework::DataRef& ref)
{
  // We do extraction on the low level to efficiently determine if the message
  // contains an object inheriting MergeInterface or TObject. If we did it the
  // the following way and catch an exception:
  // framework::DataRefUtils::as<MergeInterface>(ref)
  // it could cause a memory leak if `ref` contained a non-owning TCollection.
  // This way we also avoid doing most of the checks twice.
  const static std::string errorPrefix = "Could not extract object to be merged: ";

  using DataHeader = o2::header::DataHeader;
  auto header = o2::header::get<const DataHeader*>(ref.header);
  if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
    throw std::runtime_error(errorPrefix + "It is not ROOT-serialized");
  }

  o2::framework::FairTMessage ftm(const_cast<char*>(ref.payload), header->payloadSize);
  auto* storedClass = ftm.GetClass();
  if (storedClass == nullptr) {
    throw std::runtime_error(errorPrefix + "Unknown stored class");
  }

  auto* mergeInterfaceClass = TClass::GetClass(typeid(MergeInterface));
  auto* tObjectClass = TClass::GetClass(typeid(TObject));

  bool inheritsFromMergeInterface = storedClass->InheritsFrom(mergeInterfaceClass);
  bool inheritsFromTObject = storedClass->InheritsFrom(tObjectClass);

  if (!inheritsFromMergeInterface && !inheritsFromTObject) {
    throw std::runtime_error(
      errorPrefix + "Class '" + storedClass->GetName() + "'does not inherit from MergeInterface nor TObject");
  }

  auto* object = ftm.ReadObjectAny(storedClass);
  if (object == nullptr) {
    throw std::runtime_error(
      errorPrefix + "Failed to read object with name '" + storedClass->GetName() + "' from message using ROOT serialization.");
  }

  if (inheritsFromTObject) {
    return TObjectPtr(static_cast<TObject*>(object), algorithm::deleteTCollections);
  } else {
    return MergeInterfacePtr(static_cast<MergeInterface*>(object));
  }
}

} // namespace object_store_helpers

} // namespace o2::mergers