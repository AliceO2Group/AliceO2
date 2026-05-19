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

#include "Framework/BigEndian.h"
#include <catch_amalgamated.hpp>
#include <cstdint>
#include <cstring>

using namespace o2::framework;

TEST_CASE("bigEndianCopy: typeSize 1 is a plain copy")
{
  alignas(64) uint8_t dest[64] = {};
  uint8_t src[4] = {0x01, 0x02, 0x03, 0x04};
  bigEndianCopy(dest, src, 4, 1);
  REQUIRE(std::memcmp(dest, src, 4) == 0);
}

TEST_CASE("bigEndianCopy: uint16 byte swap")
{
  alignas(64) uint8_t dest[64] = {};
  uint8_t src[2] = {0xCA, 0xFE}; // big-endian 0xCAFE
  bigEndianCopy(dest, src, 1, 2);
  uint16_t result;
  std::memcpy(&result, dest, 2);
  REQUIRE(result == 0xCAFE);
}

TEST_CASE("bigEndianCopy: uint32 byte swap")
{
  alignas(64) uint8_t dest[64] = {};
  uint8_t src[4] = {0xDE, 0xAD, 0xBE, 0xEF}; // big-endian 0xDEADBEEF
  bigEndianCopy(dest, src, 1, 4);
  uint32_t result;
  std::memcpy(&result, dest, 4);
  REQUIRE(result == 0xDEADBEEF);
}

TEST_CASE("bigEndianCopy: uint64 byte swap")
{
  alignas(64) uint8_t dest[64] = {};
  uint8_t src[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
  bigEndianCopy(dest, src, 1, 8);
  uint64_t result;
  std::memcpy(&result, dest, 8);
  REQUIRE(result == 0x0123456789ABCDEFULL);
}

TEST_CASE("bigEndianCopy: multiple elements")
{
  alignas(64) uint8_t dest[64] = {};
  uint8_t src[8] = {0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04};
  bigEndianCopy(dest, src, 4, 2);
  auto* p = reinterpret_cast<uint16_t*>(dest);
  REQUIRE(p[0] == 1);
  REQUIRE(p[1] == 2);
  REQUIRE(p[2] == 3);
  REQUIRE(p[3] == 4);
}
