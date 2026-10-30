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

// libFuzzer target: a whole file in memory, read through every layer. Malformed
// input must end in a FormatError; any other exception, crash, hang or
// out-of-bounds access is a bug. Seed it with small corpus files.

#include "AODNative/ArrowTable.h"
#include "AODNative/Container.h"
#include "AODNative/Errors.h"
#include "../RootFraming.h"

using namespace o2::aod::native;

namespace
{
// Decoding allocates what the file says it holds; keep that within the
// fuzzer's memory limit. The reader itself has no such limit.
constexpr int64_t kMaxDecodedBytes = 64 * 1024 * 1024;

bool small(TreeInfo const& tree)
{
  int64_t total = 0;
  for (auto& b : tree.branches) {
    if (b.entries < 0 || b.fixedLen < 0 || b.entries > kMaxDecodedBytes || b.fixedLen > 1024) {
      return false;
    }
    total += b.entries * b.fixedLen * 8;
  }
  return total < kMaxDecodedBytes;
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(uint8_t const* data, size_t size)
{
  auto source = std::make_shared<test::MemoryByteSource>(std::vector<uint8_t>(data, data + size));
  try {
    RootFile file{source};
    file.topKeys();
    file.readStringMap("metaData");
    for (auto& df : file.dataFrames()) {
      try {
        for (auto& key : file.keysOf(df)) {
          if (key.className != "TTree") {
            continue;
          }
          try {
            auto tree = file.readTree(key);
            if (small(tree)) {
              (void)readTree(*source, tree, Options{});
            }
          } catch (FormatError const&) {
          }
        }
      } catch (FormatError const&) {
      }
    }
  } catch (FormatError const&) {
  }
  return 0;
}
