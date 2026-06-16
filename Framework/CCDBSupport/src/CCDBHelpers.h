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
#ifndef O2_FRAMEWORK_CCDBHELPERS_H_
#define O2_FRAMEWORK_CCDBHELPERS_H_

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/Output.h"
#include "Headers/DataHeader.h"
#include "MemoryResources/MemoryResources.h"
#include <unordered_map>
#include <string>

namespace o2::framework
{

struct CCDBHelpers {
  struct ParserResult {
    std::unordered_map<std::string, std::string> remappings;
    std::string error;
  };
  static AlgorithmSpec fetchFromCCDB();
  static ParserResult parseRemappings(char const*);

  /// Adopt a freshly-fetched CCDB payload as a new SHM message and prune
  /// the previously cached one for this path. The new SHM message is
  /// adopted BEFORE the old cached one is pruned
  /// @a allocator producer-device DPL DataAllocator
  /// @a cache read-only view of the producer-local path -> CacheId map;
  /// @a path CCDB path
  /// @a output DPL Output matcher
  /// @a v freshly-fetched CCDB payload; consumed by the call, leaving @a v empty
  /// @a method serialization-method tag written into the message header
  /// @return the new CacheId; the caller must record it in its map
  static DataAllocator::CacheId adoptAndReplaceCachedMessage(
    DataAllocator& allocator,
    std::unordered_map<std::string, DataAllocator::CacheId> const& cache,
    std::string const& path,
    Output const& output,
    o2::pmr::vector<char>&& v,
    o2::header::SerializationMethod method);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CCDBHELPERS_H_
