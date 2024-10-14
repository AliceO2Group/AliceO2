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
#ifndef O2_FRAMEWORK_OBJECTCACHE_H_
#define O2_FRAMEWORK_OBJECTCACHE_H_

#include "Framework/DataRef.h"
#include <unordered_map>
#include <map>

namespace o2::framework
{

/// A cache for CCDB objects or objects in general
/// which have more than one timeframe of lifetime.
struct ObjectCache {
  struct Id {
    int64_t value;
    static Id fromRef(DataRef& ref)
    {
      return {reinterpret_cast<int64_t>(ref.payload)};
    }
    bool operator==(const Id& other) const
    {
      return value == other.value;
    }

    struct hash_fn {
      std::size_t operator()(const Id& id) const
      {
        return id.value;
      }
    };
  };
  /// A cache for deserialised objects.
  /// This keeps a mapping so that we can tell if a given
  /// path was already received and it's blob stored in
  /// .second.
  std::unordered_map<std::string, Id> matcherToId;
  /// A map from a CacheId (which is the void* ptr of the previous map).
  /// to an actual (type erased) pointer to the deserialised object.
  std::unordered_map<Id, void*, Id::hash_fn> idToObject;

  /// A cache to the deserialised metadata
  /// We keep it separate because we want to avoid that looking up
  /// the metadata also pollutes the object cache.
  std::unordered_map<std::string, Id> matcherToMetadataId;
  std::unordered_map<Id, std::map<std::string, std::string>, Id::hash_fn> idToMetadata;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_OBJECTCACHE_H_
