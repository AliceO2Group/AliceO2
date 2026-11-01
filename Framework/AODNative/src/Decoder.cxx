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

#include "AODNative/Decoder.h"
#include "AODNative/Codecs.h"
#include "AODNative/Errors.h"
#include "Endian.h"
#include "Cursor.h"
#include <atomic>
#include <bit>
#include <climits>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

namespace o2::aod::native
{

namespace
{
// Fixed part of a TBasket key, after which come the class name, name and title.
constexpr size_t kBasketHeaderTail = 19;

template <typename T>
void byteswapAll(uint8_t* data, size_t count)
{
  for (size_t i = 0; i < count; ++i) {
    T v;
    std::memcpy(&v, data + i * sizeof(T), sizeof(T));
    v = bswap(v);
    std::memcpy(data + i * sizeof(T), &v, sizeof(T));
  }
}
} // namespace

void byteswapInPlace(uint8_t* data, size_t count, size_t typeSize)
{
  if constexpr (std::endian::native == std::endian::big) {
    return;
  }
  switch (typeSize) {
    case 2:
      byteswapAll<uint16_t>(data, count);
      break;
    case 4:
      byteswapAll<uint32_t>(data, count);
      break;
    case 8:
      byteswapAll<uint64_t>(data, count);
      break;
  }
}

/// The value bytes of one basket, in order: either plain bytes (uncompressed,
/// embedded or mapped) or a decompressor over the compressed payload.
struct BranchDecoder::BasketValues {
  std::span<const uint8_t> plain;
  std::optional<PayloadDecompressor> compressed;
  size_t border = 0;
  size_t consumed = 0;

  void read(uint8_t* dst, size_t n)
  {
    if (compressed) {
      compressed->read(dst, n);
    } else {
      std::memcpy(dst, plain.data() + consumed, n);
    }
    consumed += n;
  }

  /// The first @a n value bytes as the only read: lets zstd decode in place.
  void readAll(uint8_t* dst, size_t n)
  {
    if (compressed) {
      compressed->readPrefix(dst, n);
      consumed += n;
    } else {
      read(dst, n);
    }
  }
};

BranchDecoder::BranchDecoder(ByteSource& source, Options const& options)
  : mSource(source), mOptions(options), mUrl(source.url())
{
}

void BranchDecoder::checkBaskets(BranchInfo const& branch) const
{
  int64_t next = 0;
  for (auto& b : branch.baskets) {
    if (b.firstEntry != next || b.nEntries < 0) {
      throw CorruptData(mUrl, b.seek, "baskets of branch " + branch.name + " do not cover its entries contiguously");
    }
    next += b.nEntries;
  }
  if (next != branch.entries) {
    throw CorruptData(mUrl, 0, "baskets of branch " + branch.name + " hold " + std::to_string(next) + " entries, but the branch has " + std::to_string(branch.entries));
  }
}

template <typename F>
void BranchDecoder::forEachBasket(BranchInfo const& branch, bool parallel, F&& fn)
{
  checkBaskets(branch);
  auto task = [this, &branch, &fn](size_t i) {
    auto const& ref = branch.baskets[i];
    BasketValues values;
    std::vector<uint8_t> storage; // only when the source cannot be mapped
    if (ref.embedded) {
      values.plain = ref.embeddedPayload;
      values.border = ref.embeddedPayload.size();
    } else {
      auto bytes = mSource.tryMap(ref.seek, ref.nbytes);
      if (bytes.empty()) {
        if (ref.seek > mSource.size() || ref.nbytes > mSource.size() - ref.seek) {
          throw CorruptData(mUrl, ref.seek, "basket of branch " + branch.name + " past the end of the file");
        }
        storage.resize(ref.nbytes);
        mSource.readInto(ref.seek, ref.nbytes, storage.data());
        bytes = storage;
      }
      Cursor c{bytes, mUrl, ref.seek};
      auto nbytes = c.u32();
      c.i16(); // key fVersion
      auto objlen = c.u32();
      c.u32(); // fDatime
      auto keylen = c.u16();
      if (nbytes != ref.nbytes || keylen < kBasketHeaderTail || keylen > nbytes) {
        c.fail("basket header of branch " + branch.name + " does not match its TBranch");
      }
      c.seek(keylen - kBasketHeaderTail);
      c.u16(); // fVersion
      c.i32(); // fBufferSize
      c.i32(); // fNevBufSize
      auto nevbuf = c.i32();
      auto last = c.i32();
      if (nevbuf != ref.nEntries || last < keylen || (uint32_t)(last - keylen) > objlen) {
        c.fail("basket header of branch " + branch.name + " does not match its TBranch");
      }
      values.border = last - keylen;
      auto payload = bytes.subspan(keylen);
      if (objlen != nbytes - keylen) {
        values.compressed.emplace(payload, objlen, mUrl, ref.seek + keylen);
      } else {
        values.plain = payload;
      }
    }
    fn(i, values);
  };

  auto n = branch.baskets.size();
  int threads = parallel ? std::min<int>(mOptions.threads, n) : 1;
  if (threads <= 1) {
    for (size_t i = 0; i < n; ++i) {
      task(i);
    }
    return;
  }
  // Baskets write disjoint slices of the destination, so no locking is needed.
  std::atomic<size_t> next = 0;
  std::exception_ptr error;
  std::mutex errorMutex;
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < n;) {
        try {
          task(i);
        } catch (...) {
          std::lock_guard lock{errorMutex};
          if (!error) {
            error = std::current_exception();
          }
          next = n;
        }
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

namespace
{
void checkNeeded(std::string const& url, BranchInfo const& branch, BasketRef const& ref, size_t needed, size_t border)
{
  if (needed > border) {
    throw CorruptData(url, ref.seek, "basket of branch " + branch.name + " starting from " + std::to_string(ref.firstEntry) + " holds " + std::to_string(border) + " bytes, but " + std::to_string(needed) + " are needed");
  }
}
} // namespace

void BranchDecoder::decodeValues(BranchInfo const& branch, uint8_t* dst, size_t nbytes)
{
  auto elementSize = leafElementSize(branch.leafClass);
  size_t rowBytes = elementSize * branch.fixedLen;
  if (nbytes != branch.entries * rowBytes) {
    throw FormatError("destination of " + std::to_string(nbytes) + " bytes for branch " + branch.name + " which needs " + std::to_string(branch.entries * rowBytes));
  }
  forEachBasket(branch, true, [&](size_t i, BasketValues& values) {
    auto const& ref = branch.baskets[i];
    size_t needed = ref.nEntries * rowBytes;
    checkNeeded(mUrl, branch, ref, needed, values.border);
    auto* out = dst + ref.firstEntry * rowBytes;
    values.readAll(out, needed);
    byteswapInPlace(out, needed / elementSize, elementSize);
  });
}

void BranchDecoder::decodeVLAValues(BranchInfo const& branch, std::span<const int32_t> offsets, uint8_t* dst, size_t nbytes)
{
  auto elementSize = leafElementSize(branch.leafClass);
  if (offsets.size() != (size_t)branch.entries + 1) {
    throw FormatError("branch " + branch.name + " has " + std::to_string(branch.entries) + " entries, but " + std::to_string(offsets.size()) + " offsets were given");
  }
  if (nbytes != (size_t)offsets.back() * elementSize) {
    throw FormatError("destination of " + std::to_string(nbytes) + " bytes for branch " + branch.name + " which needs " + std::to_string(offsets.back() * elementSize));
  }
  forEachBasket(branch, true, [&](size_t i, BasketValues& values) {
    auto const& ref = branch.baskets[i];
    auto begin = offsets[ref.firstEntry];
    auto end = offsets[ref.firstEntry + ref.nEntries];
    if (end < begin) {
      throw CorruptData(mUrl, ref.seek, "invalid offset range for branch " + branch.name);
    }
    size_t needed = (size_t)(end - begin) * elementSize;
    checkNeeded(mUrl, branch, ref, needed, values.border);
    auto* out = dst + (size_t)begin * elementSize;
    values.readAll(out, needed);
    byteswapInPlace(out, needed / elementSize, elementSize);
  });
}

void BranchDecoder::decodeBooleans(BranchInfo const& branch, uint8_t* bitmap, size_t nbytes)
{
  size_t total = branch.entries * branch.fixedLen;
  if (nbytes * 8 < total) {
    throw FormatError("bitmap of " + std::to_string(nbytes) + " bytes for " + std::to_string(total) + " booleans of branch " + branch.name);
  }
  std::memset(bitmap, 0, nbytes);
  std::vector<uint8_t> window(std::min(kBooleanWindow, total));
  // The value index is global across baskets, so a basket boundary can fall
  // inside a byte of the bitmap: decode serially.
  size_t index = 0;
  forEachBasket(branch, false, [&](size_t i, BasketValues& values) {
    auto const& ref = branch.baskets[i];
    size_t remaining = (size_t)ref.nEntries * branch.fixedLen;
    checkNeeded(mUrl, branch, ref, remaining, values.border);
    while (remaining > 0) {
      auto k = std::min(remaining, window.size());
      values.read(window.data(), k);
      for (size_t j = 0; j < k; ++j, ++index) {
        bitmap[index / 8] |= (uint8_t)((window[j] != 0) << (index % 8));
      }
      remaining -= k;
    }
  });
}

std::vector<int32_t> BranchDecoder::decodeOffsets(BranchInfo const& sizeBranch)
{
  if (sizeBranch.leafClass != 'I' || sizeBranch.fixedLen != 1) {
    throw UnsupportedLayout("size branch " + sizeBranch.name + " is not a scalar 32-bit integer");
  }
  // Decode the sizes after the leading 0, then prefix-sum in place.
  std::vector<int32_t> offsets(sizeBranch.entries + 1);
  decodeValues(sizeBranch, (uint8_t*)(offsets.data() + 1), sizeBranch.entries * sizeof(int32_t));
  int64_t total = 0;
  offsets[0] = 0;
  for (int64_t i = 1; i <= sizeBranch.entries; ++i) {
    total += (uint32_t)offsets[i];
    // Arrow lists use 32 bit offsets, a larger total can only come from corrupted sizes.
    if (total > INT32_MAX) {
      throw CorruptData(mUrl, 0, "invalid sizes for branch " + sizeBranch.name + ": offsets overflow at entry " + std::to_string(i - 1));
    }
    offsets[i] = (int32_t)total;
  }
  return offsets;
}

} // namespace o2::aod::native
