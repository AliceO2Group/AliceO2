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

// o2-test-framework-aod-native-vs-root [--threads N] [--pread] [--prefetch N] [--max-dfs N] [--bench root|native] <file>...
//
// Reads every tree of every DF with the TTree plugin and with the native
// reader, serialises both as DPL does (deferred output stream into an Arrow
// IPC stream) and requires the bytes to be identical. Exits non-zero on any
// difference, naming the first differing column. With --bench, reads with
// that backend only and reports wall time, CPU time and peak RSS instead.
// A file given as <native url>=<ROOT url> is read from different places by
// the two readers, e.g. over HTTP by one and locally by the other.

#include "../src/AODInputFile.h"
#include "../src/NativeAOD.h"
#include "AODNative/Errors.h"
#include "Framework/PluginManager.h"
#include "Framework/Plugins.h"
#include "Framework/RootArrowFilesystem.h"
#include <arrow/buffer.h>
#include <arrow/dataset/file_base.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <TGrid.h>
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>

using namespace o2::framework;

namespace
{

template <typename T>
T orThrow(arrow::Result<T> result, std::string const& what)
{
  if (!result.ok()) {
    throw std::runtime_error(what + ": " + result.status().ToString());
  }
  return std::move(result).ValueUnsafe();
}

struct Serialised {
  std::shared_ptr<arrow::Buffer> bytes;
  std::string capability;
};

// Same steps as DataInputDescriptor::readTree, FragmentToBatch and
// DataAllocator::adopt, with a heap buffer in place of the shm message.
Serialised serialise(RootObjectReadingFactory& factory, std::shared_ptr<VirtualRootFileSystemBase> fs, std::string const& path)
{
  arrow::dataset::FileSource source{path, fs};
  for (auto& capability : factory.capabilities) {
    if (!capability.getHandle(fs, capability.lfn2objectPath(path))) {
      continue;
    }
    auto format = capability.factory().format();
    auto physicalSchema = orThrow(format->Inspect(source), "inspect");
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto& field : physicalSchema->fields()) {
      if (!field->name().ends_with("_size")) {
        fields.push_back(field);
      }
    }
    auto datasetSchema = std::make_shared<arrow::Schema>(fields);
    auto fragment = orThrow(format->MakeFragment(source, {}, physicalSchema), "fragment");
    auto options = std::make_shared<arrow::dataset::ScanOptions>();
    options->dataset_schema = datasetSchema;
    auto batch = orThrow(orThrow(format->ScanBatchesAsync(options, fragment), "scan")().result(), "scan");

    auto mock = std::make_shared<arrow::io::MockOutputStream>();
    (void)arrow::ipc::MakeStreamWriter(mock.get(), batch->schema()).ValueOrDie()->WriteRecordBatch(*batch);
    std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0, 64).ValueOrDie();
    if (!buffer->Reserve(mock->Tell().ValueOrDie()).ok()) {
      throw std::runtime_error("cannot reserve " + path);
    }
    auto stream = capability.factory().deferredOutputStreamer(fragment, buffer);
    auto writer = arrow::ipc::MakeStreamWriter(stream, batch->schema()).ValueOrDie();
    auto status = writer->WriteRecordBatch(*batch);
    if (status.ok()) {
      status = writer->Close();
    }
    if (status.ok()) {
      status = stream->Close();
    }
    if (!status.ok()) {
      throw std::runtime_error(path + ": " + status.ToString());
    }
    return {buffer, capability.name};
  }
  throw std::runtime_error("no capability reads " + path);
}

std::shared_ptr<arrow::RecordBatch> decode(std::shared_ptr<arrow::Buffer> const& bytes)
{
  auto reader = arrow::ipc::RecordBatchStreamReader::Open(std::make_shared<arrow::io::BufferReader>(bytes)).ValueOrDie();
  std::shared_ptr<arrow::RecordBatch> batch;
  (void)reader->ReadNext(&batch);
  return batch;
}

std::string describeDifference(Serialised const& root, Serialised const& native)
{
  auto r = decode(root.bytes);
  auto n = decode(native.bytes);
  if (!r || !n) {
    return "one of the two streams has no batch";
  }
  if (!r->schema()->Equals(*n->schema())) {
    return "schemas differ:\n  root:   " + r->schema()->ToString() + "\n  native: " + n->schema()->ToString();
  }
  if (r->num_rows() != n->num_rows()) {
    return "rows differ: " + std::to_string(r->num_rows()) + " vs " + std::to_string(n->num_rows());
  }
  for (int i = 0; i < r->num_columns(); ++i) {
    if (!r->column(i)->Equals(n->column(i))) {
      return "column " + r->schema()->field(i)->name() + " differs";
    }
  }
  return "same values, different bytes (" + std::to_string(root.bytes->size()) + " vs " + std::to_string(native.bytes->size()) + ")";
}

double cpuSeconds(timeval const& tv) { return tv.tv_sec + tv.tv_usec * 1e-6; }

// The DFs of the file in on-disk order, at most @a maxDFs of them if not negative.
std::vector<std::string> dataFramesOf(AODInputFile& file, int maxDFs)
{
  static const std::regex dfRegex{"DF_[0-9]+"};
  std::vector<std::string> dfs;
  for (auto& key : file.topLevelKeys()) {
    if (std::regex_match(key.name, dfRegex) && (maxDFs < 0 || (int)dfs.size() < maxDFs) && std::ranges::find(dfs, key.name) == dfs.end()) {
      dfs.push_back(key.name);
    }
  }
  return dfs;
}

} // namespace

int main(int argc, char** argv)
{
  o2::aod::native::Options nativeOptions;
  int maxDFs = -1;
  std::string bench;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      nativeOptions.threads = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--pread") == 0) {
      nativeOptions.localIO = o2::aod::native::Options::LocalIO::PRead;
    } else if (std::strcmp(argv[i], "--prefetch") == 0 && i + 1 < argc) {
      nativeOptions.prefetchDepth = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max-dfs") == 0 && i + 1 < argc) {
      maxDFs = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
      bench = argv[++i];
    } else {
      files.push_back(argv[i]);
    }
  }
  if (files.empty() || (!bench.empty() && bench != "root" && bench != "native")) {
    std::fprintf(stderr, "usage: %s [--threads N] [--pread] [--prefetch N] [--max-dfs N] [--bench root|native] <file>...\n", argv[0]);
    return 2;
  }

  RootObjectReadingFactory factory;
  std::vector<LoadablePlugin> plugins;
  for (auto spec : {"O2FrameworkAnalysisNativeAODSupport:NativeAODObjectReadingCapability", "O2Framework:TTreeObjectReadingCapability"}) {
    for (auto& plugin : PluginManager::parsePluginSpecString(spec)) {
      plugins.push_back(plugin);
    }
  }
  PluginManager::loadFromPlugin<RootObjectReadingCapability, RootObjectReadingCapabilityPlugin>(plugins, factory.capabilities);
  for (auto& name : files) {
    if (name.starts_with("alien://") && !gGrid) {
      TGrid::Connect("alien://");
    }
  }

  int failures = 0;
  if (!bench.empty()) {
    for (auto& name : files) {
      auto start = std::chrono::steady_clock::now();
      rusage before;
      getrusage(RUSAGE_SELF, &before);
      std::shared_ptr<AODInputFile> file;
      if (bench == "root") {
        file = RootAODInputFile::open(name, factory);
      } else {
        file = openNativeAOD(name, factory, nativeOptions);
      }
      size_t trees = 0, bytes = 0;
      auto order = dataFramesOf(*file, maxDFs);
      file->setReadOrder(order);
      int dfs = order.size();
      for (auto& df : order) {
        for (auto& tree : file->keysIn(df)) {
          bytes += serialise(factory, file->filesystem(), df + "/" + tree).bytes->size();
          ++trees;
        }
      }
      auto bytesRead = file->bytesRead();
      auto readCalls = file->readCalls();
      file.reset();
      double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      rusage after;
      getrusage(RUSAGE_SELF, &after);
      double user = cpuSeconds(after.ru_utime) - cpuSeconds(before.ru_utime);
      double sys = cpuSeconds(after.ru_stime) - cpuSeconds(before.ru_stime);
#ifdef __APPLE__
      double maxRssMiB = after.ru_maxrss / 1048576.;
#else
      double maxRssMiB = after.ru_maxrss / 1024.;
#endif
      std::printf("%s %s: %d DFs, %zu trees, %.1f MB out, read %.1f MB in %llu calls; wall %.2f s, user %.2f s, sys %.2f s, %.0f MB/s out, max RSS %.0f MiB\n",
                  bench.c_str(), name.c_str(), dfs, trees, bytes / 1e6, bytesRead / 1e6, (unsigned long long)readCalls,
                  wall, user, sys, bytes / 1e6 / wall, maxRssMiB);
    }
    return 0;
  }
  for (auto& file : files) {
    auto split = file.find('=');
    auto name = file.substr(0, split);
    auto rootName = split == std::string::npos ? name : file.substr(split + 1);
    auto root = RootAODInputFile::open(rootName, factory);
    if (!root) {
      std::printf("FAIL %s: ROOT cannot open it\n", name.c_str());
      ++failures;
      continue;
    }
    std::shared_ptr<AODInputFile> native;
    try {
      native = openNativeAOD(name, factory, nativeOptions);
    } catch (o2::aod::native::FormatError const& e) {
      std::printf("FAIL %s: native reader cannot open it: %s\n", name.c_str(), e.what());
      ++failures;
      continue;
    }
    if (root->parentFiles() != native->parentFiles() || root->metaData() != native->metaData()) {
      std::printf("FAIL %s: parentFiles or metaData differ\n", name.c_str());
      ++failures;
    }
    size_t trees = 0, rootBytes = 0, nativeBytes = 0;
    auto order = dataFramesOf(*root, maxDFs);
    native->setReadOrder(order);
    int dfs = order.size();
    for (auto& df : order) {
      auto nativeKeys = native->keysIn(df);
      if (nativeKeys != root->keysIn(df)) {
        std::printf("FAIL %s: keys of %s differ\n", name.c_str(), df.c_str());
        ++failures;
        continue;
      }
      for (auto& tree : nativeKeys) {
        auto path = df + "/" + tree;
        try {
          auto r = serialise(factory, root->filesystem(), path);
          auto n = serialise(factory, native->filesystem(), path);
          ++trees;
          rootBytes += r.bytes->size();
          nativeBytes += n.bytes->size();
          if (r.capability != "ttree" || n.capability != "native-ttree") {
            std::printf("FAIL %s %s: read by %s and %s\n", name.c_str(), path.c_str(), r.capability.c_str(), n.capability.c_str());
            ++failures;
          } else if (!r.bytes->Equals(*n.bytes)) {
            std::printf("FAIL %s %s: %s\n", name.c_str(), path.c_str(), describeDifference(r, n).c_str());
            ++failures;
          }
        } catch (std::exception const& e) {
          std::printf("FAIL %s %s: %s\n", name.c_str(), path.c_str(), e.what());
          ++failures;
        }
      }
    }
    std::printf("%s: %d DFs, %zu trees, %zu / %zu bytes serialised (root / native), native read %llu bytes in %llu calls\n",
                name.c_str(), dfs, trees, rootBytes, nativeBytes, (unsigned long long)native->bytesRead(), (unsigned long long)native->readCalls());
  }
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
