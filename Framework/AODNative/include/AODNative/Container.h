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

#ifndef O2_FRAMEWORK_AODNATIVE_CONTAINER_H_
#define O2_FRAMEWORK_AODNATIVE_CONTAINER_H_

#include "AODNative/ByteSource.h"
#include "AODNative/Objects.h"
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace o2::aod::native
{

struct FileHeader {
  int32_t version;
  int32_t begin;
  uint64_t end, seekFree;
  int32_t nbytesFree, nfree, nbytesName;
  uint8_t units;
  int32_t compress;
  uint64_t seekInfo;
  int32_t nbytesInfo;
};

struct KeyInfo {
  uint32_t nbytes;
  int16_t version;
  uint32_t objlen;
  uint16_t keylen, cycle;
  uint64_t seekKey, seekPdir;
  std::string className, name, title;
  bool compressed() const { return objlen != nbytes - keylen; }
};

struct DirectoryInfo {
  int16_t version;
  int32_t nbytesKeys, nbytesName;
  uint64_t seekDir, seekParent, seekKeys;
};

/// The container layer: file header, keys and directories. Only reads what
/// is asked for, so opening a file costs a handful of small reads.
class RootFile
{
 public:
  explicit RootFile(std::shared_ptr<ByteSource>);
  FileHeader const& header() const { return mHeader; }
  ByteSource& source() { return *mSource; }
  std::shared_ptr<ByteSource> const& sourcePtr() const { return mSource; }
  /// All the keys of the top directory, every cycle, in on-disk order.
  const std::vector<KeyInfo>& topKeys();
  /// DF_* directory names, deduplicated, ordered numerically, or by parent
  /// file path when a parentFiles map exists. Same order as DataInputDirector.
  std::vector<std::string> dataFrames();
  /// The keys inside DF @a df, highest cycle only, read once per DF. Unless
  /// the file is mapped, the stored bytes of its TTrees are fetched along, in
  /// one read: each read is a round trip for a remote file.
  std::vector<KeyInfo> keysOf(const std::string& df);
  /// The TTrees whose bytes keysOf() fetches along; all of them when empty.
  void setTreesToPrefetch(std::vector<std::string> names) { mTreesToPrefetch = std::move(names); }
  TreeInfo readTree(const KeyInfo&);
  /// Empty map if the key does not exist. Read once per key.
  std::map<std::string, std::string> readStringMap(const std::string& key);
  /// The uncompressed payload of a key.
  std::vector<uint8_t> readPayload(const KeyInfo&);

 private:
  KeyInfo parseKey(std::span<const uint8_t> buf, uint64_t fileOffset);
  DirectoryInfo readDirectory(uint64_t offset);
  std::vector<KeyInfo> readKeyList(DirectoryInfo const&);
  std::optional<KeyInfo> findTopKey(std::string const& name);
  std::vector<uint8_t> readStored(const KeyInfo&);

  std::shared_ptr<ByteSource> mSource;
  std::string mUrl;
  FileHeader mHeader;
  std::optional<std::vector<KeyInfo>> mTopKeys;
  // Stored bytes of the TTrees of the last DF listed, by key offset, until read.
  std::map<uint64_t, std::vector<uint8_t>> mPrefetched;
  std::map<std::string, std::vector<KeyInfo>> mKeysOf;
  std::vector<std::string> mTreesToPrefetch;
  std::map<std::string, std::map<std::string, std::string>> mStringMaps;
};

/// Keeps only the highest cycle of each name, preserving first-seen order.
std::vector<KeyInfo> highestCycles(std::vector<KeyInfo> const& keys);

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_CONTAINER_H_
