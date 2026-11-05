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

// o2-aod-native-inspect <file> [--json]
//
// Prints the file header, every top-level key and, per DF and tree, the
// branch layout and basket locations, using only the ROOT-free reader.

#include "AODNative/Container.h"
#include "AODNative/Errors.h"
#include <fmt/format.h>
#include <cstdio>
#include <cstring>
#include <string>

using namespace o2::aod::native;

namespace
{
std::string quote(std::string const& s)
{
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += (char)c;
    } else if (c < 0x20) {
      out += fmt::format("\\u{:04x}", c);
    } else {
      out += (char)c;
    }
  }
  return out + "\"";
}

char const* boolean(bool b)
{
  return b ? "true" : "false";
}

void printJSON(RootFile& file)
{
  auto& h = file.header();
  fmt::print("{{\n");
  fmt::print(R"(  "header": {{"version": {}, "begin": {}, "end": {}, "seekFree": {}, "nbytesFree": {}, "nfree": {}, )"
             R"("nbytesName": {}, "units": {}, "compress": {}, "seekInfo": {}, "nbytesInfo": {}}},)"
             "\n",
             h.version, h.begin, h.end, h.seekFree, h.nbytesFree, h.nfree, h.nbytesName, (int)h.units, h.compress, h.seekInfo, h.nbytesInfo);
  fmt::print(R"(  "keys": [)");
  char const* sep = "\n";
  for (auto& k : file.topKeys()) {
    fmt::print(R"({}    {{"name": {}, "class": {}, "cycle": {}, "seek": {}, "nbytes": {}, "objlen": {}}})",
               sep, quote(k.name), quote(k.className), k.cycle, k.seekKey, k.nbytes, k.objlen);
    sep = ",\n";
  }
  fmt::print("\n  ],\n");
  fmt::print(R"(  "dataFrames": [)");
  char const* dfSep = "\n";
  for (auto& df : file.dataFrames()) {
    fmt::print(R"({}    {{"name": {}, "trees": [)", dfSep, quote(df));
    dfSep = ",\n";
    char const* treeSep = "\n";
    for (auto& key : file.keysOf(df)) {
      if (key.className != "TTree") {
        continue;
      }
      auto tree = file.readTree(key);
      fmt::print(R"({}      {{"name": {}, "entries": {}, "totBytes": {}, "zipBytes": {}, "branches": [)",
                 treeSep, quote(tree.name), tree.entries, tree.totBytes, tree.zipBytes);
      treeSep = ",\n";
      char const* branchSep = "\n";
      for (auto& b : tree.branches) {
        fmt::print(R"({}        {{"name": {}, "leaf": "TLeaf{}", "len": {}, "isUnsigned": {}, "countBranch": {}, )"
                   R"("entries": {}, "totBytes": {}, "zipBytes": {}, "baskets": [)",
                   branchSep, quote(b.name), b.leafClass, b.fixedLen, boolean(b.isUnsigned), quote(b.countBranch),
                   b.entries, b.totBytes, b.zipBytes);
        branchSep = ",\n";
        char const* basketSep = "";
        for (auto& basket : b.baskets) {
          fmt::print("{}[{}, {}, {}, {}, {}]", basketSep, basket.seek, basket.nbytes, basket.firstEntry, basket.nEntries,
                     boolean(basket.embedded));
          basketSep = ", ";
        }
        fmt::print("]}}");
      }
      fmt::print("\n      ]}}");
    }
    fmt::print("\n    ]}}");
  }
  fmt::print("\n  ]\n}}\n");
}

void printText(RootFile& file)
{
  auto& h = file.header();
  fmt::print("ROOT file version {}, begin {}, end {}, compression {}, {} seeks\n", h.version, h.begin, h.end, h.compress,
             h.version >= 1000000 ? "64-bit" : "32-bit");
  fmt::print("\n{:<32} {:<16} {:>5} {:>14} {:>10} {:>10}\n", "key", "class", "cycle", "seek", "nbytes", "objlen");
  for (auto& k : file.topKeys()) {
    fmt::print("{:<32} {:<16} {:>5} {:>14} {:>10} {:>10}\n", k.name, k.className, k.cycle, k.seekKey, k.nbytes, k.objlen);
  }
  for (auto& df : file.dataFrames()) {
    fmt::print("\n{}\n", df);
    for (auto& key : file.keysOf(df)) {
      if (key.className != "TTree") {
        fmt::print("  {} ({}, skipped)\n", key.name, key.className);
        continue;
      }
      auto tree = file.readTree(key);
      fmt::print("  {}: {} entries, {} bytes, {} compressed\n", tree.name, tree.entries, tree.totBytes, tree.zipBytes);
      for (auto& b : tree.branches) {
        fmt::print("    {:<28} TLeaf{}{} len {}{}{}, {} baskets:", b.name, b.leafClass, b.isUnsigned ? " unsigned" : "",
                   b.fixedLen, b.countBranch.empty() ? "" : ", count ", b.countBranch, b.baskets.size());
        for (auto& basket : b.baskets) {
          if (basket.embedded) {
            fmt::print(" [embedded {}+{}]", basket.firstEntry, basket.nEntries);
          } else {
            fmt::print(" [{}:{} {}+{}]", basket.seek, basket.nbytes, basket.firstEntry, basket.nEntries);
          }
        }
        fmt::print("\n");
      }
    }
  }
}
} // namespace

int main(int argc, char** argv)
{
  std::string path;
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) {
      json = true;
    } else if (path.empty()) {
      path = argv[i];
    } else {
      path.clear();
      break;
    }
  }
  if (path.empty()) {
    fmt::print(stderr, "usage: {} <file> [--json]\n", argv[0]);
    return 2;
  }
  try {
    RootFile file{ByteSource::open(path, Options{})};
    if (json) {
      printJSON(file);
    } else {
      printText(file);
    }
  } catch (FormatError const& e) {
    std::fflush(stdout);
    fmt::print(stderr, "{}: {}\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
