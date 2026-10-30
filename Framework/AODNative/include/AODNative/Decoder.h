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

#ifndef O2_FRAMEWORK_AODNATIVE_DECODER_H_
#define O2_FRAMEWORK_AODNATIVE_DECODER_H_

#include "AODNative/ByteSource.h"
#include "AODNative/Objects.h"
#include <cstdint>
#include <span>
#include <vector>

namespace o2::aod::native
{

/// Byteswaps @a count big-endian values of @a typeSize bytes in place.
void byteswapInPlace(uint8_t* data, size_t count, size_t typeSize);

/// Decodes the baskets of a branch straight into caller-owned memory (a
/// shared memory slice, or a heap buffer in tools and tests). Values are
/// decompressed into the destination, stopping at the end of each basket's
/// value region, and byteswapped there: no intermediate decode buffer.
class BranchDecoder
{
 public:
  /// Size of the boolean window, the only decode-side scratch memory.
  static constexpr size_t kBooleanWindow = 64 * 1024;

  BranchDecoder(ByteSource& source, Options const& options);
  std::string const& url() const { return mUrl; }

  /// Values of a scalar or fixed-size array branch, @a dst holding exactly
  /// entries × fixedLen × element size bytes.
  void decodeValues(BranchInfo const& branch, uint8_t* dst, size_t nbytes);
  /// Values of a variable-length array branch. @a offsets are the entries + 1
  /// offsets from decodeOffsets() on its count branch.
  void decodeVLAValues(BranchInfo const& branch, std::span<const int32_t> offsets, uint8_t* dst, size_t nbytes);
  /// Booleans (one byte each on disk) bit-packed into @a bitmap, which is zeroed
  /// first. @a nbytes is the full size of the bitmap.
  void decodeBooleans(BranchInfo const& branch, uint8_t* bitmap, size_t nbytes);
  /// Decodes a <col>_size branch into entries + 1 int32 offsets.
  std::vector<int32_t> decodeOffsets(BranchInfo const& sizeBranch);

 private:
  struct BasketValues;
  /// Calls @a fn(basket index, reader) for each basket, where the reader yields
  /// the value bytes of that basket in order. Baskets are independent, so this
  /// fans out over threads when @a parallel is set and options allow it.
  template <typename F>
  void forEachBasket(BranchInfo const& branch, bool parallel, F&& fn);
  void checkBaskets(BranchInfo const& branch) const;

  ByteSource& mSource;
  Options mOptions;
  std::string mUrl;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_DECODER_H_
