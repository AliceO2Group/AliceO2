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