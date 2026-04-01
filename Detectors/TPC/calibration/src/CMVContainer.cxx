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
#include <queue>
#include <fmt/format.h>

#include "TFile.h"

#include "TPCCalibration/CMVContainer.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

int32_t CMVPerTF::cmvToSigned(uint16_t raw)
{
  const int32_t mag = raw & 0x7FFF;
  return (raw >> 15) ? mag : -mag;
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

uint16_t CMVPerTFCompressed::signedToCmv(int32_t val)
{
  const uint16_t mag = static_cast<uint16_t>(std::abs(val)) & 0x7FFF;
  return static_cast<uint16_t>((val >= 0 ? 0x8000u : 0u) | mag);
}

int32_t CMVPerTFCompressed::zigzagDecode(uint32_t value)
{
  return static_cast<int32_t>((value >> 1) ^ -(value & 1));
}

uint32_t CMVPerTFCompressed::decodeVarint(const uint8_t*& data, const uint8_t* end)
{
  uint32_t value = 0;
  int shift = 0;
  while (data < end && (*data & 0x80)) {
    value |= static_cast<uint32_t>(*data & 0x7F) << shift;
    shift += 7;
    ++data;
  }
  if (data >= end) {
    throw std::runtime_error("CMVPerTFCompressed::decompress: unexpected end of compressed data");
  }
  value |= static_cast<uint32_t>(*data) << shift;
  ++data;
  return value;
}

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
  const bool positive = (raw >> 15) & 1;          // bit 15: sign (1=positive, 0=negative)
  const float magnitude = (raw & 0x7FFF) / 128.f; // lower 15 bits, shift right by 7 (divide by 2^7)
  return positive ? magnitude : -magnitude;
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

void CMVPerTF::applyDynamicPrecision(uint16_t steps)
{
  if (steps == 0) {
    return;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF; ++i) {
    const uint16_t raw = mDataPerTF[i];
    if (raw == 0) {
      continue;
    }
    const uint16_t rounded = static_cast<uint16_t>(((raw & 0x7FFFu) + 64u) >> 7); // round(|float value|) to nearest integer
    if (rounded > steps) {
      continue; // above range: keep full precision
    }
    // rounded <= steps: store as nearest integer
    // rounded=0 (|v| < 0.5 ADC): store as exact zero
    // rounded>0: store sign * rounded
    mDataPerTF[i] = (rounded == 0) ? 0 : static_cast<uint16_t>((raw & 0x8000u) | (rounded << 7));
  }
}

CMVPerTFCompressed CMVPerTF::compress() const
{
  CMVPerTFCompressed out;
  out.firstOrbit = firstOrbit;
  out.firstBC = firstBC;
  out.mCompressedData.reserve(static_cast<size_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF);

  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    int32_t prev = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
      const int32_t val = cmvToSigned(mDataPerTF[cru * cmv::NTimeBinsPerTF + tb]);
      const int32_t delta = val - prev;
      prev = val;
      encodeVarintInto(zigzagEncode(delta), out.mCompressedData);
    }
  }
  return out;
}

CMVPerTFHuffman CMVPerTF::compressHuffman() const
{
  CMVPerTFHuffman out;
  out.firstOrbit = firstOrbit;
  out.firstBC = firstBC;

  // Step 1: compute zigzag-encoded deltas
  const uint32_t total = static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF;
  std::vector<uint32_t> zigzags;
  zigzags.reserve(total);
  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    int32_t prev = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
      const int32_t val = cmvToSigned(mDataPerTF[cru * cmv::NTimeBinsPerTF + tb]);
      zigzags.push_back(zigzagEncode(val - prev));
      prev = val;
    }
  }

  // Step 2: count symbol frequencies
  std::map<uint32_t, uint64_t> freq;
  for (const uint32_t z : zigzags) {
    ++freq[z];
  }

  // Step 3: build Huffman tree using index-based min-heap
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

  // min-heap comparator: by frequency, then by symbol for determinism
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

  // Step 4: assign code lengths via iterative DFS
  std::map<uint32_t, uint8_t> codeLens;
  {
    const int root = heap[0];
    std::vector<std::pair<int, int>> stack; // (node_idx, depth)
    stack.push_back({root, 0});
    while (!stack.empty()) {
      auto [idx, depth] = stack.back();
      stack.pop_back();
      if (nodes[idx].isLeaf) {
        codeLens[nodes[idx].sym] = static_cast<uint8_t>(depth == 0 ? 1 : depth); // single-symbol edge case
      } else {
        stack.push_back({nodes[idx].left, depth + 1});
        stack.push_back({nodes[idx].right, depth + 1});
      }
    }
  }

  // Step 5: sort by (codeLen ASC, symbol ASC) for canonical assignment
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
  std::map<uint32_t, std::pair<uint32_t, uint8_t>> codeTable; // sym -> (code, len)
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

  // Step 6: serialise table header into mHuffmanData
  auto& buf = out.mHuffmanData;
  buf.reserve(4 + symLens.size() * 5 + 8 + (zigzags.size() / 8 + 1));

  const uint32_t numSym = static_cast<uint32_t>(symLens.size());
  buf.push_back(static_cast<uint8_t>(numSym & 0xFF));
  buf.push_back(static_cast<uint8_t>((numSym >> 8) & 0xFF));
  buf.push_back(static_cast<uint8_t>((numSym >> 16) & 0xFF));
  buf.push_back(static_cast<uint8_t>((numSym >> 24) & 0xFF));
  for (const auto& sl : symLens) {
    buf.push_back(static_cast<uint8_t>(sl.sym & 0xFF));
    buf.push_back(static_cast<uint8_t>((sl.sym >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((sl.sym >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((sl.sym >> 24) & 0xFF));
    buf.push_back(sl.len);
  }

  // Placeholder for totalBits (8 bytes), filled in after encoding
  const size_t totalBitsOffset = buf.size();
  for (int i = 0; i < 8; ++i) {
    buf.push_back(0);
  }

  // Step 7: encode bitstream (MSB-first within each byte)
  uint64_t totalBits = 0;
  uint8_t curByte = 0;
  int bitsInByte = 0;
  for (const uint32_t z : zigzags) {
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

  // Step 8: backfill totalBits
  for (int i = 0; i < 8; ++i) {
    buf[totalBitsOffset + i] = static_cast<uint8_t>((totalBits >> (8 * i)) & 0xFF);
  }

  return out;
}

CMVPerTFSparse CMVPerTF::compressSparse() const
{
  CMVPerTFSparse out;
  out.firstOrbit = firstOrbit;
  out.firstBC = firstBC;

  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    // count non-zero entries for this CRU
    uint32_t count = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
      if (mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] != 0) {
        ++count;
      }
    }
    encodeVarintInto(count, out.mSparseData);

    uint32_t prevTB = 0;
    bool first = true;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
      const uint16_t val = mDataPerTF[cru * cmv::NTimeBinsPerTF + tb];
      if (val == 0) {
        continue;
      }
      // first entry: store absolute timeBin; subsequent: store tb - prevTB
      const uint32_t delta = first ? tb : (tb - prevTB);
      encodeVarintInto(delta, out.mSparseData);
      out.mSparseData.push_back(static_cast<uint8_t>(val & 0xFF));
      out.mSparseData.push_back(static_cast<uint8_t>(val >> 8));
      prevTB = tb;
      first = false;
    }
  }
  return out;
}

void CMVPerTFHuffman::decompress(CMVPerTF* cmv) const
{
  if (!cmv) {
    throw std::invalid_argument("CMVPerTFHuffman::decompress: cmv pointer is null");
  }
  cmv->firstOrbit = firstOrbit;
  cmv->firstBC = firstBC;
  std::fill(std::begin(cmv->mDataPerTF), std::end(cmv->mDataPerTF), uint16_t(0));

  // Local helpers
  auto zigzagDecode = [](uint32_t value) -> int32_t {
    return static_cast<int32_t>((value >> 1) ^ -(value & 1));
  };
  auto signedToCmv = [](int32_t val) -> uint16_t {
    const uint16_t mag = static_cast<uint16_t>(std::abs(val)) & 0x7FFF;
    return static_cast<uint16_t>((val >= 0 ? 0x8000u : 0u) | mag);
  };

  const uint8_t* ptr = mHuffmanData.data();
  const uint8_t* end = ptr + mHuffmanData.size();

  auto readU32 = [&]() -> uint32_t {
    if (ptr + 4 > end) {
      throw std::runtime_error("CMVPerTFHuffman::decompress: unexpected end reading uint32");
    }
    const uint32_t v = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
                       (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
    ptr += 4;
    return v;
  };

  // Read symbol table
  const uint32_t numSym = readU32();
  struct SymLen {
    uint32_t sym;
    uint8_t len;
  };
  std::vector<SymLen> symLens(numSym);
  for (uint32_t i = 0; i < numSym; ++i) {
    symLens[i].sym = readU32();
    if (ptr >= end) {
      throw std::runtime_error("CMVPerTFHuffman::decompress: unexpected end reading code length");
    }
    symLens[i].len = *ptr++;
  }
  // symLens is in canonical order (sorted by len ASC, sym ASC at compress time)

  // Reconstruct firstCode[len] and symsByLen[len] for canonical Huffman decode
  std::map<uint8_t, uint32_t> firstCode;              // len -> first canonical code at that length
  std::map<uint8_t, std::vector<uint32_t>> symsByLen; // len -> symbols in canonical order
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

  // Read totalBits
  if (ptr + 8 > end) {
    throw std::runtime_error("CMVPerTFHuffman::decompress: unexpected end reading totalBits");
  }
  uint64_t totalBits = 0;
  for (int i = 0; i < 8; ++i) {
    totalBits |= static_cast<uint64_t>(ptr[i]) << (8 * i);
  }
  ptr += 8;

  // Decode bitstream
  const uint8_t* bsPtr = ptr;
  uint8_t curByte = 0;
  int bitPos = -1; // MSB=7 down to LSB=0; -1 triggers fresh byte load

  auto nextBit = [&]() -> int {
    if (bitPos < 0) {
      if (bsPtr >= end) {
        throw std::runtime_error("CMVPerTFHuffman::decompress: unexpected end of bitstream");
      }
      curByte = *bsPtr++;
      bitPos = 7;
    }
    const int bit = (curByte >> bitPos) & 1;
    --bitPos;
    return bit;
  };

  const uint8_t minLen = symLens.empty() ? 1 : symLens.front().len;
  const uint8_t maxLen = symLens.empty() ? 1 : symLens.back().len;
  const uint32_t totalSymbols = static_cast<uint32_t>(CRU::MaxCRU) * cmv::NTimeBinsPerTF;
  uint64_t bitsRead = 0;

  // Decode all symbols into a flat array, then reconstruct mDataPerTF with delta decoding
  std::vector<uint32_t> zigzags;
  zigzags.reserve(totalSymbols);

  while (zigzags.size() < totalSymbols) {
    uint32_t accum = 0;
    bool found = false;
    for (uint8_t curLen = 1; curLen <= maxLen; ++curLen) {
      if (bitsRead >= totalBits) {
        throw std::runtime_error("CMVPerTFHuffman::decompress: bitstream exhausted before all symbols decoded");
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
          zigzags.push_back(sv[idx]);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      throw std::runtime_error("CMVPerTFHuffman::decompress: invalid Huffman code in bitstream");
    }
  }

  // Reconstruct mDataPerTF from zigzag-encoded deltas
  uint32_t s = 0;
  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    int32_t prev = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb, ++s) {
      const int32_t val = prev + zigzagDecode(zigzags[s]);
      prev = val;
      cmv->mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] = signedToCmv(val);
    }
  }
}

uint32_t CMVPerTFSparse::decodeVarint(const uint8_t*& data, const uint8_t* end)
{
  uint32_t value = 0;
  int shift = 0;
  while (data < end && (*data & 0x80)) {
    value |= static_cast<uint32_t>(*data & 0x7F) << shift;
    shift += 7;
    ++data;
  }
  if (data >= end) {
    throw std::runtime_error("CMVPerTFSparse::decompress: unexpected end of sparse data");
  }
  value |= static_cast<uint32_t>(*data) << shift;
  ++data;
  return value;
}

void CMVPerTFSparse::decompress(CMVPerTF* cmv) const
{
  if (!cmv) {
    throw std::invalid_argument("CMVPerTFSparse::decompress: cmv pointer is null");
  }
  cmv->firstOrbit = firstOrbit;
  cmv->firstBC = firstBC;
  std::fill(std::begin(cmv->mDataPerTF), std::end(cmv->mDataPerTF), uint16_t(0));

  const uint8_t* ptr = mSparseData.data();
  const uint8_t* end = ptr + mSparseData.size();

  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    const uint32_t count = decodeVarint(ptr, end);
    uint32_t tb = 0;
    bool first = true;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t delta = decodeVarint(ptr, end);
      tb = first ? delta : (tb + delta);
      first = false;
      if (ptr + 2 > end) {
        throw std::runtime_error("CMVPerTFSparse::decompress: unexpected end of sparse data reading value");
      }
      const uint16_t val = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
      ptr += 2;
      cmv->mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] = val;
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
  const uint8_t* ptr = mCompressedData.data();
  const uint8_t* end = ptr + mCompressedData.size();

  for (int cru = 0; cru < static_cast<int>(CRU::MaxCRU); ++cru) {
    int32_t prev = 0;
    for (uint32_t tb = 0; tb < cmv::NTimeBinsPerTF; ++tb) {
      const int32_t val = prev + zigzagDecode(decodeVarint(ptr, end));
      prev = val;
      cmv->mDataPerTF[cru * cmv::NTimeBinsPerTF + tb] = signedToCmv(val);
    }
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