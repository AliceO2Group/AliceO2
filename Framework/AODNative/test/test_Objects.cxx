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

#include "AODNative/Errors.h"
#include "AODNative/Objects.h"
#include "Cursor.h"
#include "RootFraming.h"
#include <catch_amalgamated.hpp>

using namespace o2::aod::native;
using namespace o2::aod::native::test;

TEST_CASE("Cursor reads big-endian integers")
{
  std::vector<uint8_t> buf{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  std::string file = "memory";
  Cursor c{buf, file};
  CHECK(c.u16() == 0x1234);
  CHECK(c.u32() == 0x56789abc);
  CHECK(c.u8() == 0xde);
  CHECK(c.u64() == 0xf001020304050607ULL);
  CHECK(c.remaining() == 0);
  CHECK_THROWS_AS(c.u8(), CorruptData);
}

TEST_CASE("Cursor reads both TString encodings")
{
  BigEndianWriter w;
  w.string("short");
  w.bytes.push_back(0xFF);
  w.put<uint32_t>(300);
  w.bytes.insert(w.bytes.end(), 300, 'x');
  std::string file = "memory";
  Cursor c{w.bytes, file};
  CHECK(c.string() == "short");
  CHECK(c.string() == std::string(300, 'x'));
  CHECK(c.remaining() == 0);
}

TEST_CASE("Cursor never reads out of bounds")
{
  std::vector<uint8_t> buf{0x05, 'a', 'b'}; // TString claiming 5 bytes, only 2 there
  std::string file = "memory";
  Cursor c{buf, file};
  CHECK_THROWS_AS(c.string(), CorruptData);
  Cursor d{buf, file};
  CHECK_THROWS_AS(d.seek(4), CorruptData);
  CHECK_THROWS_AS(d.skip(4), CorruptData);
  Cursor e{buf, file};
  CHECK_THROWS_AS(e.cstring(), CorruptData);
}

TEST_CASE("Cursor reports displacement relative to the key")
{
  std::vector<uint8_t> buf(10);
  std::string file = "memory";
  Cursor c{buf, file, 1000, 64};
  c.skip(4);
  CHECK(c.displacement() == 68);
}

TEST_CASE("truncated tree payloads raise errors, never crash")
{
  // Arbitrary bytes, then every truncation of them: each must fail cleanly.
  std::vector<uint8_t> junk(200);
  for (size_t i = 0; i < junk.size(); ++i) {
    junk[i] = (uint8_t)(i * 37 + 11);
  }
  // A plausible start: byte count with the flag set, TTree version 20.
  junk[0] = 0x40;
  junk[4] = 0x00;
  junk[5] = 20;
  for (size_t n = 0; n <= junk.size(); n += 7) {
    std::span<const uint8_t> part{junk.data(), n};
    CHECK_THROWS_AS(readTreePayload(part, 64, "memory", 0), FormatError);
  }
}

TEST_CASE("unknown class versions raise UnsupportedVersion")
{
  BigEndianWriter w;
  w.put<uint32_t>(0x40000000 | 100); // byte count
  w.put<uint16_t>(19);               // TTree version 19: not supported
  w.bytes.resize(104);
  CHECK_THROWS_AS(readTreePayload(w.bytes, 64, "memory", 0), UnsupportedVersion);
}

TEST_CASE("leaf element sizes")
{
  CHECK(leafElementSize('O') == 1);
  CHECK(leafElementSize('B') == 1);
  CHECK(leafElementSize('S') == 2);
  CHECK(leafElementSize('I') == 4);
  CHECK(leafElementSize('F') == 4);
  CHECK(leafElementSize('L') == 8);
  CHECK(leafElementSize('D') == 8);
  CHECK_THROWS_AS(leafElementSize('C'), UnsupportedLayout);
}
