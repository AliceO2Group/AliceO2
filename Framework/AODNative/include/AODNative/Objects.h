// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_AODNATIVE_OBJECTS_H_
#define O2_FRAMEWORK_AODNATIVE_OBJECTS_H_

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace o2::aod::native
{

/// Where the bytes of one basket are. For an embedded basket (stored inside
/// the TBranch rather than as its own key) the uncompressed value bytes are
/// kept in @a embeddedPayload and @a seek / @a nbytes are 0.
struct BasketRef {
  uint64_t seek;
  uint32_t nbytes;
  int64_t firstEntry;
  int32_t nEntries;
  bool embedded;
  std::vector<uint8_t> embeddedPayload;
};

/// One flat branch with a single scalar leaf. @a leafClass is the letter of
/// the TLeaf subclass (O, B, S, I, L, F, D). @a countBranch is the name of the
/// branch holding the sizes for a variable-length array, empty otherwise.
struct BranchInfo {
  std::string name;
  char leafClass;
  bool isUnsigned;
  int32_t fixedLen;
  std::string countBranch;
  int64_t entries, totBytes, zipBytes;
  std::vector<BasketRef> baskets;
};

struct TreeInfo {
  std::string name;
  int64_t entries, totBytes, zipBytes;
  std::vector<BranchInfo> branches;
};

/// Size in bytes of one value of a leaf class on disk.
size_t leafElementSize(char leafClass);

/// The supported versions of each class we deserialize. Extend it only with
/// versions found in the test corpus, each with its own layout function.
std::map<std::string, std::vector<int>> const& supportedClassVersions();

/// Deserializes a TTree from the uncompressed payload of its key. @a keylen is
/// needed because object references are relative to the start of the key.
TreeInfo readTreePayload(std::span<const uint8_t> payload, uint16_t keylen, std::string const& file, uint64_t fileOffset);

/// Deserializes a TMap of TObjString to TObjString (metaData, parentFiles).
std::map<std::string, std::string> readStringMapPayload(std::span<const uint8_t> payload, uint16_t keylen, std::string const& file, uint64_t fileOffset);

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_OBJECTS_H_
