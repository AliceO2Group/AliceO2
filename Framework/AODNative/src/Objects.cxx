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

#include "AODNative/Objects.h"
#include "AODNative/Errors.h"
#include "Cursor.h"
#include <algorithm>
#include <memory>
#include <variant>

namespace o2::aod::native
{

namespace
{
constexpr uint32_t kByteCountMask = 0x40000000;
constexpr uint32_t kClassMask = 0x80000000;
constexpr uint32_t kNewClassTag = 0xFFFFFFFF;
constexpr uint16_t kStreamedMemberWise = 0x4000;
constexpr uint32_t kIsReferenced = 1 << 4;
constexpr int64_t kMapOffset = 2;
// Fixed part of a TBasket key, after which come the class name, name and title.
constexpr size_t kBasketHeaderTail = 19;

struct LeafInfo {
  std::string name;
  std::string className;
  int32_t len = 0;
  bool isUnsigned = false;
  std::string countLeaf;
};

struct EmbeddedBasket {
  int32_t nEntries = 0;
  std::vector<uint8_t> data;
};

using Object = std::variant<std::monostate,
                            std::shared_ptr<BranchInfo>,
                            std::shared_ptr<LeafInfo>,
                            std::shared_ptr<EmbeddedBasket>,
                            std::shared_ptr<std::string>>;

template <typename T>
std::shared_ptr<T> as(Object const& o)
{
  auto* p = std::get_if<std::shared_ptr<T>>(&o);
  return p ? *p : nullptr;
}

/// Hardcoded streamers for the handful of classes an AO2D tree is made of.
/// Each layout is checked against the supported versions, and the byte count
/// of every object is used to skip what we do not need.
class ObjectReader
{
 public:
  explicit ObjectReader(Cursor& c) : mC(c) {}

  TreeInfo tree();
  std::map<std::string, std::string> stringMap();

 private:
  struct Versioned {
    size_t end;
    int version;
  };

  Versioned header(char const* className);
  void checkVersion(char const* className, int version);
  void tobject();
  std::string tnamed();
  void skipObject(char const* className, bool checkVersion = true);
  std::vector<Object> objArray();
  Object anyObject();
  Object readByClass(std::string const& className, int64_t mapAt);
  std::shared_ptr<BranchInfo> tbranch(int64_t mapAt);
  std::shared_ptr<LeafInfo> tleaf(std::string const& className, int64_t mapAt);
  std::shared_ptr<EmbeddedBasket> tbasket(int64_t mapAt);
  std::shared_ptr<std::string> tobjstring(int64_t mapAt);

  Cursor& mC;
  std::map<int64_t, std::string> mClasses;
  std::map<int64_t, Object> mObjects;
};

void ObjectReader::checkVersion(char const* className, int version)
{
  auto& table = supportedClassVersions();
  auto it = table.find(className);
  if (it == table.end() || std::find(it->second.begin(), it->second.end(), version) == it->second.end()) {
    throw UnsupportedVersion(className, version, mC.file());
  }
}

ObjectReader::Versioned ObjectReader::header(char const* className)
{
  auto start = mC.pos();
  auto bcnt = mC.u32();
  if (!(bcnt & kByteCountMask)) {
    throw UnsupportedLayout(std::string("object of class ") + className + " without byte count");
  }
  size_t end = start + 4 + (bcnt & ~kByteCountMask);
  if (end > mC.buffer().size()) {
    mC.fail(std::string("byte count of ") + className + " past the end of the buffer");
  }
  auto version = mC.u16();
  if (version & kStreamedMemberWise) {
    throw UnsupportedLayout(std::string("memberwise streaming of ") + className);
  }
  checkVersion(className, version);
  return {end, version};
}

void ObjectReader::tobject()
{
  auto version = mC.u16();
  if (version & kStreamedMemberWise) {
    // Written with a byte count: those were its high bytes. Skip the low ones.
    mC.u16();
    version = mC.u16();
  }
  checkVersion("TObject", version);
  mC.u32(); // fUniqueID
  auto bits = mC.u32();
  if (bits & kIsReferenced) {
    mC.u16(); // pidf
  }
}

std::string ObjectReader::tnamed()
{
  auto v = header("TNamed");
  tobject();
  auto name = mC.string();
  mC.string(); // fTitle
  mC.seek(v.end);
  return name;
}

void ObjectReader::skipObject(char const* className, bool check)
{
  auto start = mC.pos();
  auto bcnt = mC.u32();
  if (!(bcnt & kByteCountMask)) {
    throw UnsupportedLayout(std::string("object of class ") + className + " without byte count");
  }
  auto version = mC.u16();
  if (check) {
    checkVersion(className, version);
  }
  mC.seek(start + 4);
  mC.skip(bcnt & ~kByteCountMask);
}

std::vector<Object> ObjectReader::objArray()
{
  auto v = header("TObjArray");
  tobject();
  mC.string(); // fName
  auto size = mC.i32();
  mC.i32(); // fLowerBound
  if (size < 0) {
    mC.fail("negative TObjArray size");
  }
  std::vector<Object> result;
  for (int i = 0; i < size; ++i) {
    result.push_back(anyObject());
  }
  mC.seek(v.end);
  return result;
}

// ROOT's read_object_any scheme. Recorded offsets are buffer displacements
// (relative to the key start) plus kMapOffset.
Object ObjectReader::anyObject()
{
  auto beg = mC.displacement();
  auto begPos = mC.pos();
  auto bcnt = mC.u32();
  if (!(bcnt & kByteCountMask) || bcnt == kNewClassTag) {
    // No byte count: the word is the tag itself. Only references to objects
    // already read are written like this; new objects without a byte count
    // come from very old files.
    auto tag = bcnt;
    if (tag & kClassMask) {
      throw UnsupportedLayout("new object without byte count");
    }
    if (tag == 0) {
      return {};
    }
    if (tag == 1) {
      throw UnsupportedLayout("reference to the parent object");
    }
    auto it = mObjects.find(tag);
    if (it == mObjects.end()) {
      mC.fail("dangling object reference " + std::to_string(tag));
    }
    return it->second;
  }
  size_t end = begPos + 4 + (bcnt & ~kByteCountMask);
  if (end > mC.buffer().size()) {
    mC.fail("object byte count past the end of the buffer");
  }
  auto tagAt = mC.displacement();
  auto tag = mC.u32();
  Object result;
  if (!(tag & kClassMask)) {
    throw UnsupportedLayout("object reference with byte count");
  } else if (tag == kNewClassTag) {
    auto className = mC.cstring();
    mClasses[tagAt + kMapOffset] = className;
    result = readByClass(className, beg + kMapOffset);
  } else {
    auto it = mClasses.find(tag & ~kClassMask);
    if (it == mClasses.end()) {
      mC.fail("dangling class reference " + std::to_string(tag & ~kClassMask));
    }
    result = readByClass(it->second, beg + kMapOffset);
  }
  mC.seek(end);
  return result;
}

Object ObjectReader::readByClass(std::string const& className, int64_t mapAt)
{
  if (className == "TBranch") {
    return tbranch(mapAt);
  }
  if (className.size() == 6 && className.starts_with("TLeaf") && std::string_view("OBSILFD").find(className[5]) != std::string_view::npos) {
    return tleaf(className, mapAt);
  }
  if (className == "TBasket") {
    return tbasket(mapAt);
  }
  if (className == "TObjString") {
    return tobjstring(mapAt);
  }
  throw UnsupportedLayout("objects of class " + className);
}

std::shared_ptr<LeafInfo> ObjectReader::tleaf(std::string const& className, int64_t mapAt)
{
  // ROOT maps an object before streaming its members.
  auto leaf = std::make_shared<LeafInfo>();
  mObjects[mapAt] = leaf;
  auto vx = header(className.c_str());
  auto vl = header("TLeaf");
  leaf->className = className;
  leaf->name = tnamed();
  leaf->len = mC.i32();
  mC.i32(); // fLenType
  mC.i32(); // fOffset
  mC.u8();  // fIsRange
  leaf->isUnsigned = mC.u8() != 0;
  auto count = anyObject();
  if (auto countLeaf = as<LeafInfo>(count)) {
    leaf->countLeaf = countLeaf->name;
  } else if (!std::holds_alternative<std::monostate>(count)) {
    throw UnsupportedLayout("fLeafCount of " + leaf->name + " is not a leaf");
  }
  mC.seek(vl.end);
  // fMinimum and fMaximum, in the type of the leaf.
  mC.seek(vx.end);
  return leaf;
}

std::shared_ptr<EmbeddedBasket> ObjectReader::tbasket(int64_t mapAt)
{
  // TBasket has no versioned header of its own: it starts with its TKey.
  auto basket = std::make_shared<EmbeddedBasket>();
  mObjects[mapAt] = basket;
  auto start = mC.pos();
  mC.i32(); // fNbytes
  mC.i16(); // key fVersion
  mC.i32(); // fObjlen
  mC.u32(); // fDatime
  auto keylen = mC.u16();
  mC.u16(); // fCycle
  if (keylen < kBasketHeaderTail) {
    mC.fail("embedded basket with key length " + std::to_string(keylen));
  }
  mC.seek(start + keylen - kBasketHeaderTail);
  checkVersion("TBasket", mC.u16() % 1000);
  mC.i32(); // fBufferSize
  mC.i32(); // fNevBufSize
  auto nevbuf = mC.i32();
  auto last = mC.i32();
  int flag = mC.u8();
  if (nevbuf < 0 || last < keylen) {
    mC.fail("inconsistent embedded basket header");
  }
  basket->nEntries = nevbuf;
  if (flag == 99) {
    // Header only, no data.
    return basket;
  }
  bool displacements = flag > 40;
  if (displacements) {
    flag -= 40;
  }
  bool hasBuffer = flag > 10;
  if (hasBuffer) {
    flag -= 10;
  }
  if (flag == 1 && nevbuf > 0) {
    // Entry offsets (and displacements), each a count followed by fNevBuf ints.
    mC.skip(4 + 4 * (size_t)nevbuf);
    if (displacements) {
      mC.skip(4 + 4 * (size_t)nevbuf);
    }
  }
  if (hasBuffer) {
    // The buffer starts with a copy of the key, then the values.
    mC.skip(keylen);
    auto data = mC.bytes(last - keylen);
    basket->data.assign(data.begin(), data.end());
  }
  return basket;
}

std::shared_ptr<std::string> ObjectReader::tobjstring(int64_t mapAt)
{
  auto s = std::make_shared<std::string>();
  mObjects[mapAt] = s;
  auto v = header("TObjString");
  tobject();
  *s = mC.string();
  mC.seek(v.end);
  return s;
}

std::shared_ptr<BranchInfo> ObjectReader::tbranch(int64_t mapAt)
{
  auto b = std::make_shared<BranchInfo>();
  mObjects[mapAt] = b;
  auto v = header("TBranch");
  b->name = tnamed();
  skipObject("TAttFill", false);
  mC.i32(); // fCompress
  mC.i32(); // fBasketSize
  mC.i32(); // fEntryOffsetLen
  auto writeBasket = mC.i32();
  mC.i64(); // fEntryNumber
  skipObject("ROOT::TIOFeatures", false);
  mC.i32(); // fOffset
  auto maxBaskets = mC.i32();
  mC.i32(); // fSplitLevel
  b->entries = mC.i64();
  mC.i64(); // fFirstEntry
  b->totBytes = mC.i64();
  b->zipBytes = mC.i64();
  if (maxBaskets < 0 || writeBasket < 0 || b->entries < 0) {
    mC.fail("inconsistent TBranch header for " + b->name);
  }
  // The three basket arrays take 20 bytes per basket: check before allocating them.
  if ((size_t)maxBaskets > mC.remaining() / 20) {
    mC.fail("fMaxBaskets " + std::to_string(maxBaskets) + " of branch " + b->name + " exceeds the buffer");
  }

  if (!objArray().empty()) {
    throw UnsupportedLayout("branch " + b->name + " has sub-branches");
  }
  auto leaves = objArray();
  if (leaves.size() != 1 || !as<LeafInfo>(leaves[0])) {
    throw UnsupportedLayout("branch " + b->name + " does not have exactly one leaf");
  }
  auto baskets = objArray();

  auto readArray = [this, maxBaskets](auto reader) {
    mC.u8(); // speedbump
    std::vector<int64_t> values(maxBaskets);
    for (auto& value : values) {
      value = reader();
    }
    return values;
  };
  auto basketBytes = readArray([this] { return (int64_t)mC.i32(); });
  auto basketEntry = readArray([this] { return mC.i64(); });
  auto basketSeek = readArray([this] { return mC.i64(); });
  if (!mC.string().empty()) {
    throw UnsupportedLayout("branch " + b->name + " stores its baskets in another file");
  }
  mC.seek(v.end);

  auto leaf = as<LeafInfo>(leaves[0]);
  b->leafClass = leaf->className[5];
  b->isUnsigned = leaf->isUnsigned;
  b->fixedLen = leaf->len;
  b->countBranch = leaf->countLeaf;

  // Normal baskets end at the first empty seek, or at fWriteBasket.
  int nNormal = 0;
  while (nNormal < maxBaskets && nNormal < writeBasket && basketSeek[nNormal] != 0) {
    ++nNormal;
  }
  for (int i = 0; i < nNormal; ++i) {
    if (basketBytes[i] <= 0 || basketSeek[i] < 0) {
      mC.fail("invalid basket location in branch " + b->name);
    }
    // When fBasketEntry has no slot after the last basket, the entry count
    // is only in the basket header. The container layer fills it in (-1).
    bool hasNext = i + 1 < maxBaskets;
    if (basketEntry[i] < 0 || (hasNext && basketEntry[i + 1] < basketEntry[i])) {
      mC.fail("invalid basket entries in branch " + b->name);
    }
    int64_t n = hasNext ? basketEntry[i + 1] - basketEntry[i] : -1;
    if (n > INT32_MAX) {
      mC.fail("invalid basket entries in branch " + b->name);
    }
    b->baskets.push_back(BasketRef{(uint64_t)basketSeek[i], (uint32_t)basketBytes[i], basketEntry[i], (int32_t)n, false, {}});
  }
  // Embedded baskets exist only if the normal ones do not hold all the entries.
  bool knownNormal = nNormal < maxBaskets;
  int64_t normalEntries = nNormal == 0 ? 0 : (knownNormal ? basketEntry[nNormal] : -1);
  if (normalEntries != b->entries) {
    int64_t first = knownNormal ? normalEntries : 0;
    for (auto& o : baskets) {
      auto e = as<EmbeddedBasket>(o);
      if (!e || (!knownNormal && e->nEntries == 0)) {
        continue;
      }
      b->baskets.push_back(BasketRef{0, 0, first, e->nEntries, true, std::move(e->data)});
      first += e->nEntries;
    }
  }
  // The last normal basket holds what the others do not: computing it saves
  // reading its header, a round trip for a remote file. The decoder checks it
  // against the header anyway. If it does not add up, the container layer
  // reads the header.
  if (nNormal > 0 && b->baskets[nNormal - 1].nEntries < 0) {
    int64_t rest = b->entries - b->baskets[nNormal - 1].firstEntry;
    for (size_t i = nNormal; i < b->baskets.size(); ++i) {
      rest -= b->baskets[i].nEntries;
    }
    if (rest >= 0 && rest <= INT32_MAX) {
      b->baskets[nNormal - 1].nEntries = (int32_t)rest;
    }
  }
  return b;
}

TreeInfo ObjectReader::tree()
{
  TreeInfo t;
  auto v = header("TTree");
  t.name = tnamed();
  skipObject("TAttLine", false);
  skipObject("TAttFill", false);
  skipObject("TAttMarker", false);
  t.entries = mC.i64();
  t.totBytes = mC.i64();
  t.zipBytes = mC.i64();
  mC.skip(2 * 8); // fSavedBytes, fFlushedBytes
  mC.f64();       // fWeight
  mC.skip(4 * 4); // fTimerInterval, fScanField, fUpdate, fDefaultEntryOffsetLen
  auto nClusterRange = mC.u32();
  mC.skip(6 * 8); // fMaxEntries ... fEstimate
  mC.u8();        // speedbump
  mC.skip(8 * (size_t)nClusterRange);
  mC.u8(); // speedbump
  mC.skip(8 * (size_t)nClusterRange);
  skipObject("ROOT::TIOFeatures", false);
  for (auto& o : objArray()) {
    auto b = as<BranchInfo>(o);
    if (!b) {
      throw UnsupportedLayout("tree " + t.name + " has a branch which is not a TBranch");
    }
    t.branches.push_back(std::move(*b));
  }
  // fLeaves, fAliases, fIndexValues, fIndex, fTreeIndex, fFriends, fUserInfo, fBranchRef
  mC.seek(v.end);
  return t;
}

std::map<std::string, std::string> ObjectReader::stringMap()
{
  std::map<std::string, std::string> result;
  auto v = header("TMap");
  tobject();
  mC.string(); // fName
  auto n = mC.i32();
  for (int i = 0; i < n; ++i) {
    auto key = as<std::string>(anyObject());
    auto value = as<std::string>(anyObject());
    if (!key || !value) {
      throw UnsupportedLayout("TMap entry which is not a TObjString pair");
    }
    result[*key] = *value;
  }
  mC.seek(v.end);
  return result;
}
} // namespace

size_t leafElementSize(char leafClass)
{
  switch (leafClass) {
    case 'O':
    case 'B':
      return 1;
    case 'S':
      return 2;
    case 'I':
    case 'F':
      return 4;
    case 'L':
    case 'D':
      return 8;
  }
  throw UnsupportedLayout(std::string("leaf class TLeaf") + leafClass);
}

std::map<std::string, std::vector<int>> const& supportedClassVersions()
{
  static const std::map<std::string, std::vector<int>> table{
    {"TObject", {1}},
    {"TNamed", {1}},
    {"TObjString", {1}},
    {"TObjArray", {3}},
    {"TMap", {3}},
    {"TBasket", {3}},
    {"TTree", {20}},
    {"TBranch", {13}},
    {"TLeaf", {2}},
    {"TLeafO", {1}},
    {"TLeafB", {1}},
    {"TLeafS", {1}},
    {"TLeafI", {1}},
    {"TLeafL", {1}},
    {"TLeafF", {1}},
    {"TLeafD", {1}},
  };
  return table;
}

TreeInfo readTreePayload(std::span<const uint8_t> payload, uint16_t keylen, std::string const& file, uint64_t fileOffset)
{
  Cursor c{payload, file, fileOffset, keylen};
  return ObjectReader{c}.tree();
}

std::map<std::string, std::string> readStringMapPayload(std::span<const uint8_t> payload, uint16_t keylen, std::string const& file, uint64_t fileOffset)
{
  Cursor c{payload, file, fileOffset, keylen};
  return ObjectReader{c}.stringMap();
}

} // namespace o2::aod::native
