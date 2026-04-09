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

/// @file   CMVContainer.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch

#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <map>
#include <fmt/format.h>

#include "TFile.h"

#include "TPCCalibration/CMVContainer.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

// CMVPerTF private helpers

int32_t CMVPerTF::cmvToSigned(uint16_t raw)
{
  const int32_t mag = raw & 0x7FFF;
  return (raw >> 15) ? mag : -mag;
}

uint16_t CMVPerTF::quantizeBelowThreshold(uint16_t raw, float quantizationMean, float quantizationSigma)
{
  if (raw == 0u) {
    return raw;
  }

  if (quantizationSigma <= 0.f) {
    return raw;
  }

  const float adc = (raw & 0x7FFFu) / 128.f;
  const float distance = (adc - quantizationMean) / quantizationSigma;
  const float lossStrength = std::exp(-0.5f * distance * distance);

  // A true Gaussian bell: strongest trimming around the mean, then gradual recovery away from it
  float quantizedADC = adc;
  if (lossStrength > 0.85f) {
    quantizedADC = std::round(adc * 10.f) / 10.f;
  } else if (lossStrength > 0.60f) {
    quantizedADC = std::round(adc * 100.f) / 100.f;
  } else if (lossStrength > 0.30f) {
    quantizedADC = std::round(adc * 1000.f) / 1000.f;
  } else if (lossStrength > 0.12f) {
    quantizedADC = std::round(adc * 10000.f) / 10000.f;
  } else if (lossStrength > 0.03f) {
    quantizedADC = std::round(adc * 1000000.f) / 1000000.f;
  }

  // Snap the chosen decimal-style value back to the nearest raw I8F7 level
  const uint16_t quantizedMagnitude = static_cast<uint16_t>(std::clamp(std::lround(quantizedADC * 128.f), 0l, 0x7FFFl));
  return static_cast<uint16_t>((raw & 0x8000u) | quantizedMagnitude);
}

uint32_t CMVPerTF::zigzagEncode(int32_t value)
{
  return (static_cast<uint32_t>(value) << 1) ^ static_cast<uint32_t>(value >> 31);
}

void CMVPerTF::encodeVarintInto(uint32_t value, std::vector<uint8_t>& out)
{
  while (value > 0x7F) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

// Shared file-local helpers

namespace
{

int32_t zigzagDecodeLocal(uint32_t value)
{
  return static_cast<int32_t>((value >> 1) ^ -(value & 1));
}

uint16_t signedToCmvLocal(int32_t val)
{
  const uint16_t mag = static_cast<uint16_t>(std::abs(val)) & 0x7FFF;
  return static_cast<uint16_t>((val >= 0 ? 0x8000u : 0u) | mag);
}

uint32_t decodeVarintLocal(const uint8_t*& data, const uint8_t* end)
{
  uint32_t value = 0;
  int shift = 0;
  while (data < end && (*data & 0x80)) {
    value |= static_cast<uint32_t>(*data & 0x7F) << shift;
    shift += 7;
    ++data;
  }
  if (data >= end) {
    throw std::runtime_error("decodeVarintLocal: unexpected end of varint data");
  }
  value |= static_cast<uint32_t>(*data) << shift;
  ++data;
  return value;
}

/// Build and serialise a canonical Huffman table + bitstream over `symbols` into `buf`
/// Format:
///   4 bytes LE uint32_t : numSymbols
///   numSymbols × 5 bytes: symbol (4 bytes LE) + code length (1 byte)
///   8 bytes LE uint64_t : totalBits
///   ceil(totalBits/8) bytes: MSB-first bitstream
void huffmanEncode(const std::vector<uint32_t>& symbols, std::vector<uint8_t>& buf)
{
  // Frequency count
  std::map<uint32_t, uint64_t> freq;
  for (const uint32_t z : symbols) {
    ++freq[z];
  }

  // Build tree using index-based min-heap
  struct HNode {
    uint64_t freq{0};
    uint32_t sym{0};
    int left{-1}, right{-1};
    bool isLeaf{true};
  };
  std::vector<HNode> nodes;
  nodes.reserve(freq.size() * 2);
  for (const auto& [sym, f] : freq) {
    nodes.push_back({f, sym, -1, -1, true});
  }

  auto cmp = [&](int a, int b) {
    return nodes[a].freq != nodes[b].freq ? nodes[a].freq > nodes[b].freq : nodes[a].sym > nodes[b].sym;
  };
  std::vector<int> heap;
  heap.reserve(nodes.size());
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    heap.push_back(i);
  }
  std::make_heap(heap.begin(), heap.end(), cmp);

  while (heap.size() > 1) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    const int a = heap.back();
    heap.pop_back();
    std::pop_heap(heap.begin(), heap.end(), cmp);
    const int b = heap.back();
    heap.pop_back();
    nodes.push_back({nodes[a].freq + nodes[b].freq, 0, a, b, false});
    heap.push_back(static_cast<int>(nodes.size()) - 1);
    std::push_heap(heap.begin(), heap.end(), cmp);
  }

  // Assign code lengths via iterative DFS
  std::map<uint32_t, uint8_t> codeLens;
  {
    const int root = heap[0];
    std::vector<std::pair<int, int>> stack;
    stack.push_back({root, 0});
    while (!stack.empty()) {
      auto [idx, depth] = stack.back();
      stack.pop_back();
      if (nodes[idx].isLeaf) {
        codeLens[nodes[idx].sym] = static_cast<uint8_t>(depth == 0 ? 1 : depth);
      } else {
        stack.push_back({nodes[idx].left, depth + 1});
        stack.push_back({nodes[idx].right, depth + 1});
      }
    }
  }

  // Sort by (codeLen ASC, symbol ASC) for canonical assignment
  struct SymLen {
    uint32_t sym;
    uint8_t len;
  };
  std::vector<SymLen> symLens;
  symLens.reserve(codeLens.size());
  for (const auto& [sym, len] : codeLens) {
    symLens.push_back({sym, len});
  }
  std::sort(symLens.begin(), symLens.end(), [](const SymLen& a, const SymLen& b) {
    return a.len != b.len ? a.len < b.len : a.sym < b.sym;
  });

  // Assign canonical codes
  std::map<uint32_t, std::pair<uint32_t, uint8_t>> codeTable;
  {
    uint32_t code = 0;
    uint8_t prevLen = 0;
    for (const auto& sl : symLens) {
      if (prevLen != 0) {
        code = (code + 1) << (sl.len - prevLen);
      }
      codeTable[sl.sym] = {code, sl.len};
      prevLen = sl.len;
    }
  }

  // Serialise table header
  buf.reserve(buf.size() + 4 + symLens.size() * 5 + 8 + (symbols.size() / 8 + 1));
  const uint32_t numSym = static_cast<uint32_t>(symLens.size());
  for (int i = 0; i < 4; ++i) {
    buf.push_back(static_cast<uint8_t>((numSym >> (8 * i)) & 0xFF));
  }
  for (const auto& sl : symLens) {
    for (int i = 0; i < 4; ++i) {
      buf.push_back(static_cast<uint8_t>((sl.sym >> (8 * i)) & 0xFF));
    }
    buf.push_back(sl.len);
  }

  // Placeholder for totalBits
  const size_t totalBitsOffset = buf.size();
  for (int i = 0; i < 8; ++i) {
    buf.push_back(0);
  }

  // Encode bitstream (MSB-first)
  uint64_t totalBits = 0;
  uint8_t curByte = 0;
  int bitsInByte = 0;
  for (const uint32_t z : symbols) {
    const auto& [code, len] = codeTable.at(z);
    for (int b = static_cast<int>(len) - 1; b >= 0; --b) {
      curByte = static_cast<uint8_t>(curByte | (((code >> b) & 1u) << (7 - bitsInByte)));
      ++bitsInByte;
      ++totalBits;
      if (bitsInByte == 8) {
        buf.push_back(curByte);
        curByte = 0;
        bitsInByte = 0;
      }
    }
  }
  if (bitsInByte > 0) {
    buf.push_back(curByte);
  }

  // Backfill totalBits
  for (int i = 0; i < 8; ++i) {
    buf[totalBitsOffset + i] = static_cast<uint8_t>((totalBits >> (8 * i)) & 0xFF);
  }
}

/// Decode `N` symbols from a canonical Huffman payload at [ptr, end)
/// `ptr` must point to the start of the Huffman table header (numSymbols field)
/// After return, `ptr` is advanced past the bitstream
std::vector<uint32_t> huffmanDecode(const uint8_t*& ptr, const uint8_t* end, uint32_t N)
{
  auto readU32 = [&]() -> uint32_t {
    if (ptr + 4 > end) {
      throw std::runtime_error("huffmanDecode: unexpected end reading uint32");
    }
    const uint32_t v = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
                       (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
    ptr += 4;
    return v;
  };

  const uint32_t numSym = readU32();
  struct SymLen {
    uint32_t sym;
    uint8_t len;
  };
  std::vector<SymLen> symLens(numSym);
  for (uint32_t i = 0; i < numSym; ++i) {
    symLens[i].sym = readU32();
    if (ptr >= end) {
      throw std::runtime_error("huffmanDecode: unexpected end reading code length");
    }
    symLens[i].len = *ptr++;
  }

  std::map<uint8_t, uint32_t> firstCode;
  std::map<uint8_t, std::vector<uint32_t>> symsByLen;
  {
    uint32_t code = 0;
    uint8_t prevLen = 0;
    for (const auto& sl : symLens) {
      if (prevLen != 0) {
        code = (code + 1) << (sl.len - prevLen);
      }
      if (!firstCode.count(sl.len)) {
        firstCode[sl.len] = code;
      }
      symsByLen[sl.len].push_back(sl.sym);
      prevLen = sl.len;
    }
  }

  if (ptr + 8 > end) {
    throw std::runtime_error("huffmanDecode: unexpected end reading totalBits");
  }
  uint64_t totalBits = 0;
  for (int i = 0; i < 8; ++i) {
    totalBits |= static_cast<uint64_t>(ptr[i]) << (8 * i);
  }
  ptr += 8;

  const uint8_t minLen = symLens.empty() ? 1 : symLens.front().len;
  const uint8_t maxLen = symLens.empty() ? 1 : symLens.back().len;
  uint64_t bitsRead = 0;
  uint8_t curByte = 0;
  int bitPos = -1;

  auto nextBit = [&]() -> int {
    if (bitPos < 0) {
      if (ptr >= end) {
        throw std::runtime_error("huffmanDecode: unexpected end of bitstream");
      }
      curByte = *ptr++;
      bitPos = 7;
    }
    const int bit = (curByte >> bitPos) & 1;
    --bitPos;
    return bit;
  };

  std::vector<uint32_t> out;
  out.reserve(N);
  while (out.size() < N) {
    uint32_t accum = 0;
    bool found = false;
    for (uint8_t curLen = 1; curLen <= maxLen; ++curLen) {
      if (bitsRead >= totalBits) {
        throw std::runtime_error("huffmanDecode: bitstream exhausted before all symbols decoded");
      }
      accum = (accum << 1) | static_cast<uint32_t>(nextBit());
      ++bitsRead;
      if (curLen < minLen) {
        continue;
      }
      const auto fcIt = firstCode.find(curLen);
      if (fcIt == firstCode.end()) {
        continue;
      }
      if (accum >= fcIt->second) {
        const uint32_t idx = accum - fcIt->second;
        const auto& sv = symsByLen.at(curLen);
        if (idx < sv.size()) {
          out.push_back(sv[idx]);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      throw std::runtime_error("huffmanDecode: invalid Huffman code in bitstream");
    }
  }
  return out;
}

} // anonymous namespace

// CMVPerTF public methods

uint16_t CMVPerTF::getCMV(const int cru, const int timeBin) const
{
  if (cru < 0 || cru >= static_cast<int>(CRU::MaxCRU)) {
    throw std::out_of_range(fmt::format("CMVPerTF::getCMV: cru {} out of range [0, {})", cru, static_cast<int>(CRU::MaxCRU)));
  }
  if (timeBin < 0 || static_cast<uint32_t>(timeBin) >= cmv::NTimeBinsPerTF) {
    throw std::out_of_range(fmt::format("CMVPerTF::getCMV: timeBin {} out of range [0, {})", timeBin, static_cast<int>(cmv::NTimeBinsPerTF)));
  }
  return mDataPerTF[cru * cmv::NTimeBinsPerTF + timeBin];
}

float CMVPerTF::getCMVFloat(const int cru, const int timeBin) const
{
  const uint16_t raw = getCMV(cru, timeBin);
  const uint16_t mag = raw & 0x7FFF;
  if (mag == 0) {
    return 0.0f; // 0x0000 and 0x8000 both represent zero; return +0 to avoid -0 display
  }
  const bool positive = (raw >> 15) & 1; // bit 15: sign (1=positive, 0=negative)
  return positive ? mag / 128.f : -mag / 128.f;
}

void CMVPerTF::zeroSmallValues(float threshold)
{
  if (threshold <= 0.f) {
    return;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF; ++i) {
    const float mag = (mDataPerTF[i] & 0x7FFF) / 128.f;
    if (mag < threshold) {
      mDataPerTF[i] = 0;
    }
  }
}

void CMVPerTF::roundToIntegers(uint16_t threshold)
{
  if (threshold == 0) {
    return;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF; ++i) {
    const uint16_t raw = mDataPerTF[i];
    if (raw == 0) {
      continue;
    }
    const uint16_t rounded = static_cast<uint16_t>(((raw & 0x7FFFu) + 64u) >> 7);
    if (rounded > threshold) {
      continue; // above range: keep full precision
    }
    mDataPerTF[i] = (rounded == 0) ? 0 : static_cast<uint16_t>((raw & 0x8000u) | (rounded << 7));
  }
}

void CMVPerTF::trimGaussianPrecision(float mean, float sigma)
{
  if (sigma <= 0.f) {
    return;
  }

  for (uint32_t i = 0; i < static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF; ++i) {
    mDataPerTF[i] = quantizeBelowThreshold(mDataPerTF[i], mean, sigma);
  }
}

CMVPerTFCompressed CMVPerTF::compress(uint8_t flags) const
{
  CMVPerTFCompressed out;
  out.firstOrbit = firstOrbit;
  out.firstBC = firstBC;
  out.mFlags = flags;

  if (flags & CMVEncoding::kSparse) {
    // --- Sparse path: position stream + value stream ---

    // Single pass per CRU: build the position stream and collect raw non-zero values.
    std::vector<uint8_t> posStream;
    std::vector<uint16_t> rawValues;

    for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
      struct Entry {
        uint32_t tb;
        uint16_t val;
      };
      std::vector<Entry> entries;
      for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
        const uint16_t val = mDataPerTF[cru * cmv::NTimeBinsPerTF + tb];
        if (val != 0) {
          entries.push_back({tb, val});
        }
      }

      encodeVarintInto(static_cast<uint32_t>(entries.size()), posStream);
      uint32_t prevTB = 0;
      bool first = true;
      for (const auto& e : entries) {
        encodeVarintInto(first ? e.tb : (e.tb - prevTB), posStream);
        rawValues.push_back(e.val);
        prevTB = e.tb;
        first = false;
      }
    }

    // Encode the value stream based on flags.
    std::vector<uint8_t> valStream;
    if (flags & CMVEncoding::kZigzag) {
      std::vector<uint32_t> zigzags;
      zigzags.reserve(rawValues.size());
      for (const uint16_t v : rawValues) {
        zigzags.push_back(zigzagEncode(cmvToSigned(v)));
      }
      if (flags & CMVEncoding::kHuffman) {
        huffmanEncode(zigzags, valStream);
      } else { // kVarint
        for (const uint32_t z : zigzags) {
          encodeVarintInto(z, valStream);
        }
      }
    } else {
      // Raw uint16 LE
      for (const uint16_t v : rawValues) {
        valStream.push_back(static_cast<uint8_t>(v & 0xFF));
        valStream.push_back(static_cast<uint8_t>(v >> 8));
      }
    }

    // Assemble: [4 bytes posStreamSize][posStream][valStream]
    const uint32_t posStreamSize = static_cast<uint32_t>(posStream.size());
    out.mData.reserve(4 + posStream.size() + valStream.size());
    for (int i = 0; i < 4; ++i) {
      out.mData.push_back(static_cast<uint8_t>((posStreamSize >> (8 * i)) & 0xFF));
    }
    out.mData.insert(out.mData.end(), posStream.begin(), posStream.end());
    out.mData.insert(out.mData.end(), valStream.begin(), valStream.end());

  } else {
    // --- Dense path: all CRU * TimeBin values ---
    const uint32_t total = static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF;

    if (!(flags & CMVEncoding::kZigzag)) {
      // No encoding: raw uint16 LE
      out.mData.reserve(total * 2);
      for (uint32_t i = 0; i < total; ++i) {
        out.mData.push_back(static_cast<uint8_t>(mDataPerTF[i] & 0xFF));
        out.mData.push_back(static_cast<uint8_t>(mDataPerTF[i] >> 8));
      }
    } else {
      // Zigzag + optional delta (CRU-major, time-minor)
      const bool useDelta = (flags & CMVEncoding::kDelta) != 0;
      std::vector<uint32_t> zigzags;
      zigzags.reserve(total);
      for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
        int32_t prev = 0;
        for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
          const int32_t val = cmvToSigned(mDataPerTF[cru * cmv::NTimeBinsPerTF + tb]);
          const int32_t encoded = useDelta ? (val - prev) : val;
          if (useDelta) {
            prev = val;
          }
          zigzags.push_back(zigzagEncode(encoded));
        }
      }

      if (flags & CMVEncoding::kHuffman) {
        huffmanEncode(zigzags, out.mData);
      } else { // kVarint
        for (const uint32_t z : zigzags) {
          encodeVarintInto(z, out.mData);
        }
      }
    }
  }

  return out;
}

// CMVPerTFCompressed::decompress staged pipeline

std::vector<std::pair<int, uint32_t>> CMVPerTFCompressed::decodeSparsePositions(const uint8_t*& ptr, const uint8_t* end)
{
  // Read 4-byte LE posStreamSize
  if (ptr + 4 > end) {
    throw std::runtime_error("CMVPerTFCompressed::decompress: truncated position header");
  }
  const uint32_t posStreamSize = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
                                 (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
  ptr += 4;

  const uint8_t* posEnd = ptr + posStreamSize;
  if (posEnd > end) {
    throw std::runtime_error("CMVPerTFCompressed::decompress: posStream overflows payload");
  }

  // Decode per-CRU varint(N) + N×varint(tb_delta)
  std::vector<std::pair<int, uint32_t>> positions;
  const uint8_t* p = ptr;
  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    const uint32_t count = decodeVarintLocal(p, posEnd);
    uint32_t tb = 0;
    bool first = true;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t delta = decodeVarintLocal(p, posEnd);
      tb = first ? delta : (tb + delta);
      first = false;
      positions.emplace_back(cru, tb);
    }
  }
  ptr = posEnd; // advance past the entire position block
  return positions;
}

std::vector<uint32_t> CMVPerTFCompressed::decodeValueStream(const uint8_t*& ptr, const uint8_t* end, uint32_t N, uint8_t flags)
{
  if (flags & CMVEncoding::kHuffman) {
    // Huffman-encoded symbols
    return huffmanDecode(ptr, end, N);
  }

  if (flags & CMVEncoding::kVarint) {
    // Varint-encoded symbols
    std::vector<uint32_t> out;
    out.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
      out.push_back(decodeVarintLocal(ptr, end));
    }
    return out;
  }

  // Raw uint16 LE (no value encoding)
  std::vector<uint32_t> out;
  out.reserve(N);
  for (uint32_t i = 0; i < N; ++i) {
    if (ptr + 2 > end) {
      throw std::runtime_error("CMVPerTFCompressed::decompress: unexpected end in raw value stream");
    }
    const uint16_t v = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
    ptr += 2;
    out.push_back(v);
  }
  return out;
}

void CMVPerTFCompressed::decodeSparseValues(const std::vector<uint32_t>& symbols,
                                            const std::vector<std::pair<int, uint32_t>>& positions,
                                            uint8_t flags, CMVPerTF* cmv)
{
  const bool useZigzag = (flags & CMVEncoding::kZigzag) != 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(positions.size()); ++i) {
    uint16_t raw;
    if (useZigzag) {
      raw = signedToCmvLocal(zigzagDecodeLocal(symbols[i]));
    } else {
      raw = static_cast<uint16_t>(symbols[i]);
    }
    cmv->mDataPerTF[positions[i].first * cmv::NTimeBinsPerTF + positions[i].second] = raw;
  }
}

void CMVPerTFCompressed::decodeDenseValues(const std::vector<uint32_t>& symbols, uint8_t flags, CMVPerTF* cmv)
{
  const bool useZigzag = (flags & CMVEncoding::kZigzag) != 0;
  const bool useDelta = (flags & CMVEncoding::kDelta) != 0;

  if (!useZigzag) {
    // Symbols are raw uint16 values; write directly
    for (uint32_t i = 0; i < static_cast<uint32_t>(symbols.size()); ++i) {
      cmv->mDataPerTF[i] = static_cast<uint16_t>(symbols[i]);
    }
    return;
  }

  // Inverse zigzag + optional inverse delta (CRU-major, time-minor)
  uint32_t s = 0;
  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    int32_t prev = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb, ++s) {
      int32_t val = zigzagDecodeLocal(symbols[s]);
      if (useDelta) {
        val += prev;
        prev = val;
      }
      cmv->mDataPerTF[s] = signedToCmvLocal(val);
    }
  }
}

void CMVPerTFCompressed::decompress(CMVPerTF* cmv) const
{
  if (!cmv) {
    throw std::invalid_argument("CMVPerTFCompressed::decompress: cmv pointer is null");
  }
  cmv->firstOrbit = firstOrbit;
  cmv->firstBC = firstBC;
  std::fill(std::begin(cmv->mDataPerTF), std::end(cmv->mDataPerTF), uint16_t(0));

  const uint8_t* ptr = mData.data();
  const uint8_t* end = ptr + mData.size();

  if (mFlags & CMVEncoding::kSparse) {
    // Stage 1: decode position stream
    auto positions = decodeSparsePositions(ptr, end);
    const uint32_t N = static_cast<uint32_t>(positions.size());

    // Stage 2: decode value stream (Huffman / varint / raw)
    auto symbols = decodeValueStream(ptr, end, N, mFlags);

    // Stage 3: inverse zigzag and scatter into CMV array
    decodeSparseValues(symbols, positions, mFlags, cmv);
  } else {
    const uint32_t N = static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF;

    // Stage 1: decode value stream (Huffman / varint / raw)
    auto symbols = decodeValueStream(ptr, end, N, mFlags);

    // Stage 2: inverse zigzag, inverse delta, fill CMV array
    decodeDenseValues(symbols, mFlags, cmv);
  }
}

std::unique_ptr<TTree> CMVPerTF::toTTree() const
{
  auto tree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
  tree->SetAutoSave(0);
  tree->SetDirectory(nullptr);

  const CMVPerTF* ptr = this;
  tree->Branch("CMVPerTF", &ptr);
  tree->Fill();

  tree->ResetBranchAddresses();
  return tree;
}

std::unique_ptr<TTree> CMVPerTFCompressed::toTTree() const
{
  auto tree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
  tree->SetAutoSave(0);
  tree->SetDirectory(nullptr);

  const CMVPerTFCompressed* ptr = this;
  tree->Branch("CMVPerTFCompressed", &ptr);
  tree->Fill();

  tree->ResetBranchAddresses();
  return tree;
}

void CMVPerTF::writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree)
{
  TFile f(filename.c_str(), "RECREATE");
  if (f.IsZombie()) {
    throw std::runtime_error(fmt::format("CMVPerTF::writeToFile: cannot open '{}'", filename));
  }
  tree->Write();
  f.Close();
}

} // namespace o2::tpc
