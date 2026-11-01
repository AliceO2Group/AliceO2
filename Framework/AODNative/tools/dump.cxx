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

// o2-aod-native-dump <file> <DF_x/O2tree> <output.arrow> [--threads N] [--pread]
//
// Decodes one tree with the ROOT-free reader into heap memory and writes it
// as an Arrow IPC stream, with the dataset schema of the TTree plugin.

#include "AODNative/ArrowTable.h"
#include "AODNative/Container.h"
#include "AODNative/Errors.h"
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <cstdio>
#include <cstring>
#include <string>

using namespace o2::aod::native;

int main(int argc, char** argv)
{
  std::vector<std::string> args;
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      options.threads = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--pread") == 0) {
      options.localIO = Options::LocalIO::PRead;
    } else {
      args.push_back(argv[i]);
    }
  }
  if (args.size() != 3 || args[1].find('/') == std::string::npos) {
    std::fprintf(stderr, "usage: %s <file> <DF_x/O2tree> <output.arrow> [--threads N] [--pread]\n", argv[0]);
    return 2;
  }
  auto df = args[1].substr(0, args[1].find('/'));
  auto treeName = args[1].substr(args[1].find('/') + 1);
  try {
    RootFile file{ByteSource::open(args[0], options)};
    std::optional<TreeInfo> tree;
    for (auto& key : file.keysOf(df)) {
      if (key.name == treeName) {
        tree = file.readTree(key);
      }
    }
    if (!tree) {
      throw FormatError("no tree " + treeName + " in " + df);
    }
    auto batch = readTree(file.source(), *tree, options).ValueOrDie();
    auto out = arrow::io::FileOutputStream::Open(args[2]).ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(out, batch->schema()).ValueOrDie();
    auto status = writer->WriteRecordBatch(*batch);
    if (status.ok()) {
      status = writer->Close();
    }
    if (!status.ok()) {
      std::fprintf(stderr, "%s: %s\n", argv[0], status.ToString().c_str());
      return 1;
    }
  } catch (FormatError const& e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
