// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   pack.h
/// @author Michael Lettrich
/// @brief  packs data into a buffer

#ifndef RANS_INTERNAL_PACK_PACK_H_
#define RANS_INTERNAL_PACK_PACK_H_

#include <cstdint>
#include <cstring>
#include <array>
#include <type_traits>

#include <iostream>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/BitPtr.h"
#include "rANS/internal/pack/utils.h"

namespace o2::rans
{

template <typename storageBuffer_T = uint8_t>
[[nodiscard]] constexpr size_t computePackingBufferSize(size_t extent, size_t packingWidthBits) noexcept
{
  using namespace internal;
  using namespace utils;
  if (extent > 0) {
    constexpr size_t PackingBufferBits = toBits<packing_type>();
    const size_t packedSizeBits = extent * packingWidthBits;

    const size_t packingBufferElems = packedSizeBits / PackingBufferBits + 1;
    const size_t packingBufferBytes = packingBufferElems * sizeof(packing_type);
    const size_t storageBufferElems = utils::nBytesTo<storageBuffer_T>(packingBufferBytes);
    return storageBufferElems;
  } else {
    return 0;
  }
}

namespace internal
{

inline BitPtr packShort(BitPtr pos, uint64_t data, size_t packingWidth)
{
  assert(pos != BitPtr{});
  assert(packingWidth <= 58);
  uint8_t* posPtr = pos.toPtr<uint8_t>();
  const size_t posBitOffset = pos.getOffset<uint8_t>();

  packing_type buffer = load64(reinterpret_cast<void*>(posPtr));
  buffer |= data << posBitOffset;
  write64(posPtr, buffer);

  return pos += packingWidth;
};

inline BitPtr pack(BitPtr pos, uint64_t data, size_t packingWidth)
{
  return packShort(pos, data, packingWidth);
}

inline BitPtr packLong(BitPtr pos, uint64_t data, size_t packingWidth)
{
  assert(pos != BitPtr{});
  uint8_t* posPtr = pos.toPtr<uint8_t>();
  const size_t posBitOffset = pos.getOffset<uint8_t>();

  const size_t bitOffsetEnd = posBitOffset + packingWidth;
  packing_type buffer = load64(reinterpret_cast<void*>(posPtr));

  constexpr size_t PackingBufferBits = utils::toBits<packing_type>();

  if (bitOffsetEnd <= PackingBufferBits) {
    buffer |= data << posBitOffset;
    write64(posPtr, buffer);
  } else {
    const size_t tailBits = PackingBufferBits - posBitOffset;
    buffer |= bitExtract(data, 0, tailBits) << posBitOffset;
    write64(posPtr, buffer);
    posPtr += sizeof(packing_type);
    buffer = load64(reinterpret_cast<void*>(posPtr));
    buffer |= data >> tailBits;
    write64(posPtr, buffer);
  }

  return pos += packingWidth;
};

template <typename T>
inline T unpack(BitPtr pos, size_t packingWidth)
{
  assert(pos != BitPtr{});
  assert(packingWidth <= utils::toBits<T>());
  assert(packingWidth <= 58);

  uint8_t* posPtr = pos.toPtr<uint8_t>();
  size_t posBitOffset = pos.getOffset<uint8_t>();

  packing_type buffer = load64(reinterpret_cast<void*>(posPtr));
  return static_cast<T>(bitExtract(buffer, posBitOffset, packingWidth));
};

inline uint64_t unpackLong(BitPtr pos, size_t packingWidth)
{
  assert(pos != BitPtr{});
  assert(packingWidth < 64);

  uint8_t* posPtr = pos.toPtr<uint8_t>();
  size_t bitOffset = pos.getOffset<uint8_t>();

  constexpr size_t PackingBufferBits = utils::toBits<packing_type>();
  const size_t bitOffsetEnd = bitOffset + packingWidth;
  packing_type buffer = load64(reinterpret_cast<void*>(posPtr));
  uint64_t ret{};
  if (bitOffsetEnd <= PackingBufferBits) {
    ret = bitExtract(buffer, bitOffset, packingWidth);
  } else {
    // first part
    ret = bitExtract(buffer, bitOffset, PackingBufferBits - bitOffset);
    // second part
    bitOffset = bitOffsetEnd - PackingBufferBits;
    posPtr += sizeof(packing_type);
    buffer = load64(reinterpret_cast<void*>(posPtr));
    ret |= bitExtract(buffer, 0, bitOffset) << (packingWidth - bitOffset);
  }

  return ret;
};

template <typename input_T, typename output_T, size_t width_V>
constexpr BitPtr packStreamImpl(const input_T* __restrict inputBegin, size_t extent, output_T* outputBegin, input_T offset)
{
  assert(inputBegin != nullptr);
  assert(outputBegin != nullptr);

  constexpr size_t PackingBufferBits = utils::toBits<packing_type>();
  constexpr size_t PackingWidth = width_V;

  constexpr size_t NPackAtOnce = PackingBufferBits / PackingWidth;

  uint8_t* outputIter = reinterpret_cast<uint8_t*>(outputBegin);
  size_t outputIterBitOffset = {};

  const size_t nIterations = extent / NPackAtOnce;
  const size_t nRemainderIterations = extent % NPackAtOnce;

  auto inputIter = inputBegin;
  const auto iterEnd = inputIter + NPackAtOnce * nIterations;
  packing_type overflowBuffer{0};

  for (; inputIter < iterEnd; inputIter += NPackAtOnce) {
    packing_type packed = packMultiple<input_T, width_V>(inputIter, offset);

    const size_t tail = PackingBufferBits - outputIterBitOffset;
    overflowBuffer |= packed << outputIterBitOffset;
    outputIterBitOffset += NPackAtOnce * PackingWidth;

    if constexpr (PackingBufferBits % PackingWidth == 0) {
      write64(outputIter, overflowBuffer);
      overflowBuffer = 0;
      outputIterBitOffset = 0;
      outputIter += sizeof(packing_type);
    } else {
      if (outputIterBitOffset >= PackingBufferBits) {
        write64(outputIter, overflowBuffer);
        outputIter += sizeof(packing_type);
        overflowBuffer = packed >> tail;
      }
      outputIterBitOffset %= PackingBufferBits;
    }
  }
  write64(outputIter, overflowBuffer);

  BitPtr bitPos{outputIter, static_cast<intptr_t>(outputIterBitOffset)};
  for (size_t i = 0; i < nRemainderIterations; ++i) {
    const int64_t adjustedValue = static_cast<int64_t>(inputIter[i]) - offset;
    bitPos = pack(bitPos, adjustedValue, PackingWidth);
  }

  return bitPos;
};
} // namespace internal

template <typename input_T, typename output_T>
inline constexpr BitPtr pack(const input_T* __restrict inputBegin, size_t extent, output_T* __restrict outputBegin, size_t packingWidth, input_T offset = static_cast<input_T>(0))
{
  using namespace internal;
  using namespace utils;

  assert(inputBegin != nullptr);
  assert(outputBegin != nullptr);

  switch (packingWidth) {
    case 1:
      return packStreamImpl<input_T, output_T, 1>(inputBegin, extent, outputBegin, offset);
      break;
    case 2:
      return packStreamImpl<input_T, output_T, 2>(inputBegin, extent, outputBegin, offset);
      break;
    case 3:
      return packStreamImpl<input_T, output_T, 3>(inputBegin, extent, outputBegin, offset);
      break;
    case 4:
      return packStreamImpl<input_T, output_T, 4>(inputBegin, extent, outputBegin, offset);
      break;
    case 5:
      return packStreamImpl<input_T, output_T, 5>(inputBegin, extent, outputBegin, offset);
      break;
    case 6:
      return packStreamImpl<input_T, output_T, 6>(inputBegin, extent, outputBegin, offset);
      break;
    case 7:
      return packStreamImpl<input_T, output_T, 7>(inputBegin, extent, outputBegin, offset);
      break;
    case 8:
      return packStreamImpl<input_T, output_T, 8>(inputBegin, extent, outputBegin, offset);
      break;
    case 9:
      return packStreamImpl<input_T, output_T, 9>(inputBegin, extent, outputBegin, offset);
      break;
    case 10:
      return packStreamImpl<input_T, output_T, 10>(inputBegin, extent, outputBegin, offset);
      break;
    case 11:
      return packStreamImpl<input_T, output_T, 11>(inputBegin, extent, outputBegin, offset);
      break;
    case 12:
      return packStreamImpl<input_T, output_T, 12>(inputBegin, extent, outputBegin, offset);
      break;
    case 13:
      return packStreamImpl<input_T, output_T, 13>(inputBegin, extent, outputBegin, offset);
      break;
    case 14:
      return packStreamImpl<input_T, output_T, 14>(inputBegin, extent, outputBegin, offset);
      break;
    case 15:
      return packStreamImpl<input_T, output_T, 15>(inputBegin, extent, outputBegin, offset);
      break;
    case 16:
      return packStreamImpl<input_T, output_T, 16>(inputBegin, extent, outputBegin, offset);
      break;
    case 17:
      return packStreamImpl<input_T, output_T, 17>(inputBegin, extent, outputBegin, offset);
      break;
    case 18:
      return packStreamImpl<input_T, output_T, 18>(inputBegin, extent, outputBegin, offset);
      break;
    case 19:
      return packStreamImpl<input_T, output_T, 19>(inputBegin, extent, outputBegin, offset);
      break;
    case 20:
      return packStreamImpl<input_T, output_T, 20>(inputBegin, extent, outputBegin, offset);
      break;
    case 21:
      return packStreamImpl<input_T, output_T, 21>(inputBegin, extent, outputBegin, offset);
      break;
    case 22:
      return packStreamImpl<input_T, output_T, 22>(inputBegin, extent, outputBegin, offset);
      break;
    case 23:
      return packStreamImpl<input_T, output_T, 23>(inputBegin, extent, outputBegin, offset);
      break;
    case 24:
      return packStreamImpl<input_T, output_T, 24>(inputBegin, extent, outputBegin, offset);
      break;
    case 25:
      return packStreamImpl<input_T, output_T, 25>(inputBegin, extent, outputBegin, offset);
      break;
    case 26:
      return packStreamImpl<input_T, output_T, 26>(inputBegin, extent, outputBegin, offset);
      break;
    case 27:
      return packStreamImpl<input_T, output_T, 27>(inputBegin, extent, outputBegin, offset);
      break;
    case 28:
      return packStreamImpl<input_T, output_T, 28>(inputBegin, extent, outputBegin, offset);
      break;
    case 29:
      return packStreamImpl<input_T, output_T, 29>(inputBegin, extent, outputBegin, offset);
      break;
    case 30:
      return packStreamImpl<input_T, output_T, 30>(inputBegin, extent, outputBegin, offset);
      break;
    case 31:
      return packStreamImpl<input_T, output_T, 31>(inputBegin, extent, outputBegin, offset);
      break;
    case 32:
      return packStreamImpl<input_T, output_T, 32>(inputBegin, extent, outputBegin, offset);
      break;
    default:
      BitPtr iter{outputBegin};
      for (size_t i = 0; i < extent; ++i) {
        const int64_t adjustedValue = static_cast<int64_t>(inputBegin[i]) - offset;
        iter = packLong(iter, adjustedValue, packingWidth);
      }
      return iter;
      break;
  }
  return {};
};

template <typename input_IT, typename output_T>
inline constexpr BitPtr pack(input_IT inputBegin, size_t extent, output_T* __restrict outputBegin, size_t packingWidth,
                             typename std::iterator_traits<input_IT>::value_type offset = 0)
{
  using namespace internal;
  using namespace utils;
  using source_type = typename std::iterator_traits<input_IT>::value_type;
  input_IT inputEnd = advanceIter(inputBegin, extent);
  BitPtr outputIter{outputBegin};

  auto packImpl = [](input_IT inputBegin, input_IT inputEnd, BitPtr outputIter, source_type offset, size_t packingWidth, auto packingFunctor) -> BitPtr {
    for (auto inputIter = inputBegin; inputIter != inputEnd; ++inputIter) {
      const int64_t adjustedValue = static_cast<int64_t>(*inputIter) - offset;
      outputIter = packingFunctor(outputIter, adjustedValue, packingWidth);
    }
    return outputIter;
  };

  if (packingWidth <= 58) {
    return packImpl(inputBegin, inputEnd, outputIter, offset, packingWidth, packShort);
  } else {
    return packImpl(inputBegin, inputEnd, outputIter, offset, packingWidth, packLong);
  }
}

template <typename input_T, typename output_IT>
inline void unpack(const input_T* __restrict inputBegin, size_t extent, output_IT outputBegin, size_t packingWidth, typename std::iterator_traits<output_IT>::value_type offset = static_cast<typename std::iterator_traits<output_IT>::value_type>(0))
{
  using namespace internal;
  using namespace utils;
  using dst_type = typename std::iterator_traits<output_IT>::value_type;

  auto unpackImpl = [&](auto packer) {
    output_IT outputIt = outputBegin;
    BitPtr iter{inputBegin};
    for (size_t i = 0; i < extent; ++i) {
      auto unpacked = packer(iter, packingWidth) + offset;
      *outputIt++ = unpacked;
      iter += packingWidth;
    }
  };

  if (packingWidth <= 58) {
    unpackImpl(unpack<dst_type>);
  } else {
    unpackImpl(unpackLong);
  }
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_PACK_PACK_H_ */