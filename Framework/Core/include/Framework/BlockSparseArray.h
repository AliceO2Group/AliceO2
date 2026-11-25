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

#ifndef O2_FRAMEWORK_BLOCKSPARSEARRAY_H_
#define O2_FRAMEWORK_BLOCKSPARSEARRAY_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace o2::framework
{

/// Sparse storage of histogram bin contents (and optionally sum of squared
/// weights), keyed by a linear bin index.
///
/// The key space is cut into blocks of 2^blockBits consecutive keys. A block
/// keeps its filled bins either
///   - sparse: sorted 16-bit offsets with the contents (and sumw2) alongside, or
///   - dense:  the full arrays of the block plus a bitmap of filled bins.
/// A block turns dense once that costs at most `denseFactor` times its sparse
/// form, which bounds the memory per filled bin in both regimes. The blocks in
/// use are found through an open-addressing hash table of block ids, so there
/// is no per-bin hash table: a filled bin in a sparse block costs sizeof(T) + 2
/// bytes (+8 with sumw2), plus ~32 bytes per block in use for its descriptor.
template <typename T>
class BlockSparseArray
{
 public:
  /// Pointers to the content and sumw2 (nullptr without sumw2) of one bin.
  /// Valid until the next insertion of a bin.
  struct Ref {
    T* v = nullptr;
    double* s = nullptr;
    explicit operator bool() const { return v != nullptr; }
  };

  /// Position of an iteration over the filled bins (see next()).
  struct Cursor {
    std::vector<uint64_t> ids; // block ids in use, sorted
    size_t index = 0;
    uint32_t pos = 0;
    bool started = false;
  };

  /// keyBits: keys are < 2^keyBits. blockBits in [6, 16].
  /// The default block size, 2^12 keys, keeps the shift of a sorted block on inserting a
  /// new bin short (at most ~700 bins before the block turns dense), so filling in random
  /// order stays within ~1.5x of THnSparse; larger blocks save little memory.
  static constexpr int kDefaultBlockBits = 12;

  explicit BlockSparseArray(int keyBits, int blockBits = kDefaultBlockBits, bool sumw2 = false, double denseFactor = 4.)
    : mKeyBits(keyBits), mBlockBits(blockBits), mBlockSize(1u << blockBits), mMask((1u << blockBits) - 1), mSumw2(sumw2), mDenseFactor(denseFactor)
  {
    if (blockBits < 6 || blockBits > 16) {
      throw std::invalid_argument("BlockSparseArray: blockBits must be in [6, 16]");
    }
    if (keyBits < 0 || keyBits > 64) {
      throw std::invalid_argument("BlockSparseArray: keys must fit in 64 bits");
    }
    mPerBin = sizeof(T) + sizeof(uint16_t) + (mSumw2 ? sizeof(double) : 0);
    mDenseBytes = size_t(mBlockSize) * (sizeof(T) + (mSumw2 ? sizeof(double) : 0)) + mBlockSize / 8;
    mDenseAt = uint32_t(mDenseBytes / (denseFactor * mPerBin)) + 1;
    mDenseAt = std::min<uint32_t>(std::max<uint32_t>(mDenseAt, 8), mBlockSize);
  }
  BlockSparseArray(const BlockSparseArray&) = delete;
  BlockSparseArray& operator=(const BlockSparseArray&) = delete;
  ~BlockSparseArray() { clear(); }

  /// content[key] += w, sumw2[key] += w2 (if kept). Keys out of range are ignored.
  void add(uint64_t key, T w, double w2)
  {
    if (Ref r = ref(key, true)) {
      *r.v += w;
      if (r.s) {
        *r.s += w2;
      }
    }
  }
  void add(uint64_t key, T w = 1) { add(key, w, double(w) * double(w)); }

  /// The bin `key`: inserted (zero) if `create`, otherwise an empty Ref when not
  /// filled. Always an empty Ref for a key out of range.
  Ref ref(uint64_t key, bool create)
  {
    if (!inRange(key)) {
      return {};
    }
    Block* b = create ? &block(key >> mBlockBits) : findBlock(key >> mBlockBits);
    if (!b) {
      return {};
    }
    uint32_t off = uint32_t(key & mMask);
    if (b->cap == kDense) {
      uint64_t* bits = denseBits(*b);
      uint64_t m = uint64_t(1) << (off & 63);
      if (!(bits[off >> 6] & m)) {
        if (!create) {
          return {};
        }
        bits[off >> 6] |= m;
        ++b->n;
        ++mFilled;
      }
      return {vals(*b) + off, mSumw2 ? sw2(*b) + off : nullptr};
    }
    uint32_t i = b->n ? uint32_t(std::lower_bound(offs(*b), offs(*b) + b->n, uint16_t(off)) - offs(*b)) : 0;
    if (i < b->n && offs(*b)[i] == off) {
      return {vals(*b) + i, mSumw2 ? sw2(*b) + i : nullptr};
    }
    if (!create) {
      return {};
    }
    if (b->n == b->cap) {
      if (b->n + 1 >= mDenseAt) {
        densify(*b);
        return ref(key, true);
      }
      grow(*b);
    }
    uint16_t* o = offs(*b);
    T* v = vals(*b);
    std::memmove(o + i + 1, o + i, (b->n - i) * sizeof(uint16_t));
    std::memmove(v + i + 1, v + i, (b->n - i) * sizeof(T));
    o[i] = uint16_t(off);
    v[i] = T(0);
    double* s = nullptr;
    if (mSumw2) {
      s = sw2(*b);
      std::memmove(s + i + 1, s + i, (b->n - i) * sizeof(double));
      s[i] = 0.;
      s += i;
    }
    ++b->n;
    ++mFilled;
    return {v + i, s};
  }

  /// Content of a bin (0 if not filled or out of range).
  T get(uint64_t key) const
  {
    Ref r = const_cast<BlockSparseArray*>(this)->ref(key, false);
    return r ? *r.v : T(0);
  }
  /// Sum of squared weights of a bin (0 if not filled, not kept or out of range).
  double getSumw2(uint64_t key) const
  {
    Ref r = const_cast<BlockSparseArray*>(this)->ref(key, false);
    return r.s ? *r.s : 0.;
  }

  /// Next filled bin after `c`, in increasing key order. False at the end.
  /// No bin may be inserted while iterating.
  bool next(Cursor& c, uint64_t& key, T& v, double& s) const
  {
    if (!c.started) {
      c.ids.reserve(mUsed);
      for (const auto& slot : mSlots) {
        if (slot.id != kEmpty) {
          c.ids.push_back(slot.id);
        }
      }
      std::sort(c.ids.begin(), c.ids.end());
      c.started = true;
    }
    for (; c.index < c.ids.size(); ++c.index, c.pos = 0) {
      uint64_t id = c.ids[c.index];
      const Block* b = const_cast<BlockSparseArray*>(this)->findBlock(id);
      if (b->cap == kDense) {
        const uint64_t* bits = denseBits(*b);
        for (uint32_t off = c.pos; off < mBlockSize;) {
          uint64_t m = bits[off >> 6] >> (off & 63);
          if (!m) {
            off = (off | 63) + 1;
            continue;
          }
          off += uint32_t(std::countr_zero(m));
          key = (id << mBlockBits) + off;
          v = vals(*b)[off];
          s = mSumw2 ? sw2(*b)[off] : double(v);
          c.pos = off + 1;
          return true;
        }
      } else if (c.pos < b->n) {
        key = (id << mBlockBits) + offs(*b)[c.pos];
        v = vals(*b)[c.pos];
        s = mSumw2 ? sw2(*b)[c.pos] : double(v);
        ++c.pos;
        return true;
      }
    }
    return false;
  }

  /// f(key, content, sumw2) for every filled bin, in increasing key order.
  /// Without sumw2 the content is passed as sumw2 (Poisson errors).
  template <class F>
  void forEach(F&& f) const
  {
    Cursor c;
    uint64_t k;
    T v;
    double s;
    while (next(c, k, v, s)) {
      f(k, v, s);
    }
  }

  /// this += other (same layout).
  void merge(const BlockSparseArray& other)
  {
    if (other.mBlockBits != mBlockBits || other.mKeyBits != mKeyBits) {
      throw std::invalid_argument("BlockSparseArray::merge: incompatible layout");
    }
    other.forEach([this](uint64_t k, T v, double s) { add(k, v, s); });
  }

  /// Start keeping sum of squared weights, initialised to the contents (as THnSparse::Sumw2()).
  void enableSumw2()
  {
    if (mSumw2) {
      return;
    }
    BlockSparseArray tmp(mKeyBits, mBlockBits, true, mDenseFactor);
    forEach([&tmp](uint64_t k, T v, double) { tmp.add(k, v, double(v)); });
    swap(tmp);
  }

  bool hasSumw2() const { return mSumw2; }
  bool inRange(uint64_t key) const { return mKeyBits >= 64 || (key >> mKeyBits) == 0; }
  uint64_t filledBins() const { return mFilled; }
  /// Bytes allocated (blocks and block table), excluding malloc overhead.
  size_t bytes() const { return mBytes + mSlots.capacity() * sizeof(Slot); }
  size_t denseBlocks() const { return mDenseBlocks; }

  void clear()
  {
    for (auto& slot : mSlots) {
      if (slot.id != kEmpty) {
        std::free(slot.b.buf);
      }
    }
    mSlots.clear();
    mSlots.shrink_to_fit();
    mUsed = 0;
    mFilled = mBytes = mDenseBlocks = 0;
  }

  void swap(BlockSparseArray& o) noexcept
  {
    std::swap(mKeyBits, o.mKeyBits);
    std::swap(mBlockBits, o.mBlockBits);
    std::swap(mBlockSize, o.mBlockSize);
    std::swap(mMask, o.mMask);
    std::swap(mSumw2, o.mSumw2);
    std::swap(mDenseFactor, o.mDenseFactor);
    std::swap(mPerBin, o.mPerBin);
    std::swap(mDenseBytes, o.mDenseBytes);
    std::swap(mDenseAt, o.mDenseAt);
    mSlots.swap(o.mSlots);
    std::swap(mUsed, o.mUsed);
    std::swap(mFilled, o.mFilled);
    std::swap(mBytes, o.mBytes);
    std::swap(mDenseBlocks, o.mDenseBlocks);
  }

 private:
  static constexpr uint32_t kDense = 0xffffffffu;
  static constexpr uint64_t kEmpty = ~uint64_t(0); // no block id reaches it: ids are < 2^(64 - blockBits)

  // Layout of buf: sparse [sumw2 (double) x cap][contents (T) x cap][offsets (uint16) x cap]
  //                dense  [sumw2 (double) x S  ][contents (T) x S  ][bitmap (uint64) x S/64]
  struct Block {
    void* buf = nullptr;
    uint32_t n = 0;   // filled bins
    uint32_t cap = 0; // capacity in bins, kDense when dense
  };
  struct Slot {
    uint64_t id = kEmpty;
    Block b;
  };

  uint32_t slots(const Block& b) const { return b.cap == kDense ? mBlockSize : b.cap; }
  double* sw2(const Block& b) const { return static_cast<double*>(b.buf); }
  T* vals(const Block& b) const
  {
    return reinterpret_cast<T*>(static_cast<char*>(b.buf) + (mSumw2 ? slots(b) * sizeof(double) : 0));
  }
  uint16_t* offs(const Block& b) const { return reinterpret_cast<uint16_t*>(vals(b) + b.cap); }
  // 2^blockBits T's (a multiple of 64 of them) keep the bitmap 8-byte aligned
  uint64_t* denseBits(const Block& b) const { return reinterpret_cast<uint64_t*>(vals(b) + mBlockSize); }

  static size_t hash(uint64_t x)
  {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return size_t(x);
  }
  Block* findBlock(uint64_t id)
  {
    if (mSlots.empty()) {
      return nullptr;
    }
    size_t mask = mSlots.size() - 1;
    for (size_t i = hash(id) & mask;; i = (i + 1) & mask) {
      if (mSlots[i].id == id) {
        return &mSlots[i].b;
      }
      if (mSlots[i].id == kEmpty) {
        return nullptr;
      }
    }
  }
  /// The block `id`, created empty if needed. Rehashing moves descriptors, so
  /// a Block reference is valid only until the next block is created.
  Block& block(uint64_t id)
  {
    if (Block* b = findBlock(id)) {
      return *b;
    }
    if ((mUsed + 1) * 4 > mSlots.size() * 3) {
      rehash(mSlots.empty() ? 16 : 2 * mSlots.size());
    }
    size_t mask = mSlots.size() - 1;
    size_t i = hash(id) & mask;
    while (mSlots[i].id != kEmpty) {
      i = (i + 1) & mask;
    }
    mSlots[i].id = id;
    ++mUsed;
    return mSlots[i].b;
  }
  void rehash(size_t size)
  {
    std::vector<Slot> old(size);
    old.swap(mSlots);
    size_t mask = size - 1;
    for (const auto& slot : old) {
      if (slot.id != kEmpty) {
        size_t i = hash(slot.id) & mask;
        while (mSlots[i].id != kEmpty) {
          i = (i + 1) & mask;
        }
        mSlots[i] = slot;
      }
    }
  }

  void* allocate(size_t bytes)
  {
    void* p = std::calloc(1, bytes);
    if (!p) {
      throw std::bad_alloc();
    }
    mBytes += bytes;
    return p;
  }
  void release(Block& b, size_t bytes)
  {
    std::free(b.buf);
    mBytes -= bytes;
  }
  size_t sparseBytes(uint32_t cap) const { return size_t(cap) * mPerBin; }

  // Capacity grows 1, 2, 3, 4, 6, 9, ...: a block with a single bin (bins
  // scattered over many blocks) costs one bin, not a minimum chunk.
  void grow(Block& b)
  {
    uint32_t cap = std::min(std::max(b.cap + b.cap / 2, b.cap + 1), mDenseAt);
    Block nb;
    nb.cap = cap;
    nb.n = b.n;
    nb.buf = allocate(sparseBytes(cap));
    if (b.n) {
      std::memcpy(vals(nb), vals(b), b.n * sizeof(T));
      std::memcpy(offs(nb), offs(b), b.n * sizeof(uint16_t));
      if (mSumw2) {
        std::memcpy(sw2(nb), sw2(b), b.n * sizeof(double));
      }
    }
    if (b.buf) {
      release(b, sparseBytes(b.cap));
    }
    b = nb;
  }

  void densify(Block& b)
  {
    Block nb;
    nb.cap = kDense;
    nb.n = b.n;
    nb.buf = allocate(mDenseBytes);
    T* v = vals(nb);
    double* s = mSumw2 ? sw2(nb) : nullptr;
    uint64_t* bits = denseBits(nb);
    const uint16_t* o = offs(b);
    for (uint32_t i = 0; i < b.n; ++i) {
      v[o[i]] = vals(b)[i];
      if (s) {
        s[o[i]] = sw2(b)[i];
      }
      bits[o[i] >> 6] |= uint64_t(1) << (o[i] & 63);
    }
    release(b, sparseBytes(b.cap));
    b = nb;
    ++mDenseBlocks;
  }

  int mKeyBits;
  int mBlockBits;
  uint32_t mBlockSize;
  uint32_t mMask;
  bool mSumw2;
  double mDenseFactor;
  size_t mPerBin = 0;
  size_t mDenseBytes = 0;
  uint32_t mDenseAt = 0;
  std::vector<Slot> mSlots; // open addressing, size a power of 2, load <= 3/4
  size_t mUsed = 0;
  uint64_t mFilled = 0;
  size_t mBytes = 0, mDenseBlocks = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_BLOCKSPARSEARRAY_H_
