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

#include "AODNative/Container.h"
#include "AODNative/Codecs.h"
#include "AODNative/Errors.h"
#include "Cursor.h"
#include <algorithm>
#include <regex>
#include <unordered_set>

namespace o2::aod::native
{

namespace
{
// Big enough for the file header and for a TDirectory record (including its UUID).
constexpr size_t kHeaderReadSize = 128;
constexpr size_t kDirectoryReadSize = 64;
// Largest gap between the TTrees of a DF still read together with them.
constexpr uint64_t kMetadataGap = 4 * 1024;
// Fixed part of a TBasket key, after which come the class name, name and title.
constexpr size_t kBasketHeaderTail = 19;
// Smallest possible key header: 32-bit seeks and three empty strings.
constexpr size_t kMinKeySize = 29;
// Metadata objects (trees, maps) in AO2Ds are a few MB at most. A larger
// object length can only come from a corrupted key, so refuse to allocate it.
constexpr uint32_t kMaxMetadataSize = 256 * 1024 * 1024;

std::vector<uint8_t> readRange(ByteSource& source, uint64_t off, size_t len)
{
  // Check before allocating: the length comes from the file.
  if (off > source.size() || len > source.size() - off) {
    throw CorruptData(source.url(), off, "range of " + std::to_string(len) + " bytes past the end of the file");
  }
  std::vector<uint8_t> buf(len);
  source.readInto(off, len, buf.data());
  return buf;
}
} // namespace

RootFile::RootFile(std::shared_ptr<ByteSource> source)
  : mSource(std::move(source)),
    mUrl(mSource->url())
{
  auto buf = readRange(*mSource, 0, std::min<uint64_t>(kHeaderReadSize, mSource->size()));
  Cursor c{buf, mUrl};
  auto magic = c.bytes(4);
  if (std::memcmp(magic.data(), "root", 4) != 0) {
    throw FormatError(mUrl + " is not a ROOT file");
  }
  auto& h = mHeader;
  h.version = c.i32();
  h.begin = c.i32();
  bool big = h.version >= 1000000;
  h.end = big ? c.u64() : c.u32();
  h.seekFree = big ? c.u64() : c.u32();
  h.nbytesFree = c.i32();
  h.nfree = c.i32();
  h.nbytesName = c.i32();
  h.units = c.u8();
  h.compress = c.i32();
  h.seekInfo = big ? c.u64() : c.u32();
  h.nbytesInfo = c.i32();
  if (h.begin <= 0 || h.nbytesName <= 0) {
    c.fail("invalid file header");
  }
}

KeyInfo RootFile::parseKey(std::span<const uint8_t> buf, uint64_t fileOffset)
{
  Cursor c{buf, mUrl, fileOffset};
  KeyInfo k;
  k.nbytes = c.u32();
  k.version = c.i16();
  k.objlen = c.u32();
  c.u32(); // fDatime
  k.keylen = c.u16();
  k.cycle = c.u16();
  bool big = k.version > 1000;
  k.seekKey = big ? c.u64() : c.u32();
  k.seekPdir = big ? c.u64() : c.u32();
  k.className = c.string();
  k.name = c.string();
  k.title = c.string();
  if (k.keylen > k.nbytes || c.pos() > k.keylen) {
    c.fail("inconsistent key header for " + k.name);
  }
  return k;
}

DirectoryInfo RootFile::readDirectory(uint64_t offset)
{
  if (offset >= mSource->size()) {
    throw CorruptData(mUrl, offset, "directory record past the end of the file");
  }
  auto buf = readRange(*mSource, offset, std::min<uint64_t>(kDirectoryReadSize, mSource->size() - offset));
  Cursor c{buf, mUrl, offset};
  DirectoryInfo d;
  d.version = c.i16();
  c.u32(); // fDatimeC
  c.u32(); // fDatimeM
  d.nbytesKeys = c.i32();
  d.nbytesName = c.i32();
  bool big = d.version > 1000;
  d.seekDir = big ? c.u64() : c.u32();
  d.seekParent = big ? c.u64() : c.u32();
  d.seekKeys = big ? c.u64() : c.u32();
  return d;
}

std::vector<KeyInfo> RootFile::readKeyList(DirectoryInfo const& dir)
{
  if (dir.seekKeys == 0 || dir.nbytesKeys <= 0) {
    return {};
  }
  auto buf = readRange(*mSource, dir.seekKeys, dir.nbytesKeys);
  auto header = parseKey(buf, dir.seekKeys);
  Cursor c{buf, mUrl, dir.seekKeys};
  c.seek(header.keylen);
  auto nkeys = c.i32();
  if (nkeys < 0 || (size_t)nkeys > c.remaining() / kMinKeySize) {
    c.fail("invalid number of keys " + std::to_string(nkeys));
  }
  std::vector<KeyInfo> keys;
  keys.reserve(nkeys);
  for (int i = 0; i < nkeys; ++i) {
    auto start = c.pos();
    // The key length is only known after parsing, so parse from here to the end
    // of the list and advance by the key length.
    keys.push_back(parseKey(c.buffer().subspan(start), dir.seekKeys + start));
    c.seek(start + keys.back().keylen);
  }
  return keys;
}

const std::vector<KeyInfo>& RootFile::topKeys()
{
  if (!mTopKeys) {
    auto dir = readDirectory((uint64_t)mHeader.begin + mHeader.nbytesName);
    mTopKeys = readKeyList(dir);
  }
  return *mTopKeys;
}

std::vector<KeyInfo> highestCycles(std::vector<KeyInfo> const& keys)
{
  std::vector<KeyInfo> result;
  for (auto& k : keys) {
    auto it = std::find_if(result.begin(), result.end(), [&k](KeyInfo const& r) { return r.name == k.name; });
    if (it == result.end()) {
      result.push_back(k);
    } else if (k.cycle > it->cycle) {
      *it = k;
    }
  }
  return result;
}

std::optional<KeyInfo> RootFile::findTopKey(std::string const& name)
{
  std::optional<KeyInfo> result;
  for (auto& k : topKeys()) {
    if (k.name == name && (!result || k.cycle > result->cycle)) {
      result = k;
    }
  }
  return result;
}

std::vector<std::string> RootFile::dataFrames()
{
  static const std::regex tfRegex("/?DF_([0-9]+)(|-.*)$");
  std::vector<uint64_t> numbers;
  std::unordered_set<uint64_t> seen;
  for (auto& k : topKeys()) {
    std::smatch m;
    if (std::regex_match(k.name, m, tfRegex)) {
      auto n = std::stoull(m[1].str());
      if (seen.insert(n).second) {
        numbers.push_back(n);
      }
    }
  }
  std::sort(numbers.begin(), numbers.end());
  auto parents = readStringMap("parentFiles");
  if (!parents.empty()) {
    // Process DFs grouped by parent file, to avoid swapping between files.
    // DataInputDirector uses an unstable sort here; ties are kept in numeric order.
    auto parentOf = [&parents](uint64_t n) -> std::string const& {
      static const std::string none;
      auto it = parents.find("DF_" + std::to_string(n));
      return it == parents.end() ? none : it->second;
    };
    std::stable_sort(numbers.begin(), numbers.end(), [&](uint64_t a, uint64_t b) { return parentOf(a) < parentOf(b); });
  }
  std::vector<std::string> result;
  result.reserve(numbers.size());
  for (auto n : numbers) {
    result.push_back("DF_" + std::to_string(n));
  }
  return result;
}

std::vector<KeyInfo> RootFile::keysOf(const std::string& df)
{
  if (auto it = mKeysOf.find(df); it != mKeysOf.end()) {
    return it->second;
  }
  auto key = findTopKey(df);
  if (!key) {
    for (auto& k : topKeys()) {
      if (k.name.starts_with(df + "-")) {
        throw UnsupportedLayout("flat " + df + "-<table> layout (" + k.className + ") in " + mUrl);
      }
    }
    throw FormatError("no directory " + df + " in " + mUrl);
  }
  if (key->className != "TDirectoryFile" && key->className != "TDirectory") {
    throw UnsupportedLayout(df + " in " + mUrl + " is a " + key->className + ", not a TDirectoryFile");
  }
  if (key->compressed()) {
    throw UnsupportedLayout("compressed directory record for " + df + " in " + mUrl);
  }
  auto dir = readDirectory(key->seekKey + key->keylen);
  auto keys = highestCycles(readKeyList(dir));

  // The TTrees of a DF are written next to each other: ranges less than
  // kMetadataGap apart are read as one.
  struct Wanted {
    uint64_t offset, length, seekKey;
  };
  std::vector<Wanted> wanted;
  for (auto const& k : keys) {
    auto stored = (uint64_t)k.nbytes - k.keylen;
    auto offset = k.seekKey + k.keylen;
    bool listed = mTreesToPrefetch.empty() || std::ranges::find(mTreesToPrefetch, k.name) != mTreesToPrefetch.end();
    // Anything odd is left to readPayload(), which reports it.
    if (mSource->canMap() || !listed || k.className != "TTree" || k.nbytes < k.keylen || stored > kMaxMetadataSize ||
        offset > mSource->size() || stored > mSource->size() - offset) {
      continue;
    }
    wanted.push_back({offset, stored, k.seekKey});
  }
  std::ranges::sort(wanted, {}, &Wanted::offset);
  std::vector<std::pair<uint64_t, std::vector<uint8_t>>> spans;
  for (auto const& w : wanted) {
    if (!spans.empty()) {
      auto& [start, bytes] = spans.back();
      auto end = start + bytes.size();
      if (w.offset <= end + kMetadataGap && w.offset + w.length - start <= kMaxMetadataSize) {
        bytes.resize(std::max(end, w.offset + w.length) - start);
        continue;
      }
    }
    spans.emplace_back(w.offset, std::vector<uint8_t>(w.length));
  }
  std::vector<ReadRequest> requests;
  for (auto& [start, bytes] : spans) {
    requests.push_back({start, bytes.size(), bytes.data()});
  }
  mPrefetched.clear();
  if (!requests.empty()) {
    mSource->readvInto(requests);
  }
  size_t span = 0;
  for (auto const& w : wanted) {
    while (spans[span].first + spans[span].second.size() < w.offset + w.length) {
      ++span;
    }
    auto const* at = spans[span].second.data() + (w.offset - spans[span].first);
    mPrefetched[w.seekKey].assign(at, at + w.length);
  }
  mKeysOf.emplace(df, keys);
  return keys;
}

std::vector<uint8_t> RootFile::readStored(const KeyInfo& key)
{
  auto stored = key.nbytes - key.keylen;
  if (auto it = mPrefetched.find(key.seekKey); it != mPrefetched.end() && it->second.size() == stored) {
    auto bytes = std::move(it->second);
    mPrefetched.erase(it);
    return bytes;
  }
  return readRange(*mSource, key.seekKey + key.keylen, stored);
}

std::vector<uint8_t> RootFile::readPayload(const KeyInfo& key)
{
  auto offset = key.seekKey + key.keylen;
  auto stored = key.nbytes - key.keylen;
  if (key.objlen > kMaxMetadataSize || stored > kMaxMetadataSize) {
    throw CorruptData(mUrl, key.seekKey, "object " + key.name + " of " + std::to_string(key.objlen) + " bytes is too large for metadata");
  }
  if (!key.compressed()) {
    return readStored(key);
  }
  auto compressed = readStored(key);
  std::vector<uint8_t> result(key.objlen);
  decompressBounded(compressed, key.objlen, result.data(), result.size(), mUrl, offset);
  return result;
}

TreeInfo RootFile::readTree(const KeyInfo& key)
{
  if (key.className != "TTree") {
    throw UnsupportedLayout(key.name + " in " + mUrl + " is a " + key.className + ", not a TTree");
  }
  auto payload = readPayload(key);
  auto tree = readTreePayload(payload, key.keylen, mUrl, key.seekKey + key.keylen);
  // When fBasketEntry has no slot past the last normal basket, its entry count
  // is only in the basket header (fNevBuf).
  for (auto& branch : tree.branches) {
    for (auto& basket : branch.baskets) {
      if (basket.nEntries >= 0) {
        continue;
      }
      auto buf = readRange(*mSource, basket.seek, std::min<uint64_t>(basket.nbytes, kHeaderReadSize * 4));
      auto k = parseKey(buf, basket.seek);
      Cursor c{buf, mUrl, basket.seek};
      c.seek(k.keylen - kBasketHeaderTail);
      c.u16(); // fVersion
      c.i32(); // fBufferSize
      c.i32(); // fNevBufSize
      basket.nEntries = c.i32();
    }
  }
  return tree;
}

std::map<std::string, std::string> RootFile::readStringMap(const std::string& name)
{
  if (auto it = mStringMaps.find(name); it != mStringMaps.end()) {
    return it->second;
  }
  auto key = findTopKey(name);
  if (!key) {
    return mStringMaps[name];
  }
  if (key->className != "TMap") {
    throw UnsupportedLayout(name + " in " + mUrl + " is a " + key->className + ", not a TMap");
  }
  auto payload = readPayload(*key);
  return mStringMaps[name] = readStringMapPayload(payload, key->keylen, mUrl, key->seekKey + key->keylen);
}

} // namespace o2::aod::native
