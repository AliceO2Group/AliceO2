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

// o2-test-framework-aod-native-memory [--threads N] [--pread] <file>...
//
// Checks that decoding a DF uses bounded private memory. Every heap
// allocation (ours and the codecs') is tracked by interposing malloc. Columns
// are decoded into MAP_SHARED anonymous memory, standing in for the shared
// memory DPL provides, so the output itself is not private.
//
// Per DF, private memory is allowed for: parsed metadata, VLA offsets, and
// per worker a constant decode scratch (boolean window plus codec state). The
// decode phase must stay within threads × kDecodeScratchBound, whatever the
// size of the baskets. In pread mode the compressed input of each basket in
// flight is allowed on top.

#include "AODNative/ArrowTable.h"
#include "AODNative/Container.h"
#include "AODNative/Errors.h"
#include <arrow/memory_pool.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <malloc.h>
#include <string>
#include <sys/mman.h>

using namespace o2::aod::native;

// ---- malloc interposition ---------------------------------------------------

extern "C" void* __libc_malloc(size_t);
extern "C" void* __libc_calloc(size_t, size_t);
extern "C" void* __libc_realloc(void*, size_t);
extern "C" void* __libc_memalign(size_t, size_t);
extern "C" void __libc_free(void*);

namespace
{
std::atomic<int64_t> gCurrent{0};
std::atomic<int64_t> gPeak{0};
std::atomic<int64_t> gLargest{0};

void track(void* p)
{
  if (!p) {
    return;
  }
  int64_t n = malloc_usable_size(p);
  auto now = gCurrent.fetch_add(n) + n;
  for (auto peak = gPeak.load(); now > peak && !gPeak.compare_exchange_weak(peak, now);) {
  }
  for (auto largest = gLargest.load(); n > largest && !gLargest.compare_exchange_weak(largest, n);) {
  }
}

void untrack(void* p)
{
  if (p) {
    gCurrent.fetch_sub(malloc_usable_size(p));
  }
}
} // namespace

extern "C" {
void* malloc(size_t n)
{
  auto* p = __libc_malloc(n);
  track(p);
  return p;
}
void* calloc(size_t a, size_t b)
{
  auto* p = __libc_calloc(a, b);
  track(p);
  return p;
}
void* realloc(void* old, size_t n)
{
  untrack(old);
  auto* p = __libc_realloc(old, n);
  track(p ? p : (n ? nullptr : p));
  return p;
}
void free(void* p)
{
  untrack(p);
  __libc_free(p);
}
void* memalign(size_t a, size_t n)
{
  auto* p = __libc_memalign(a, n);
  track(p);
  return p;
}
void* aligned_alloc(size_t a, size_t n) { return memalign(a, n); }
int posix_memalign(void** out, size_t a, size_t n)
{
  *out = memalign(a, n);
  return *out ? 0 : ENOMEM;
}
}

// ---- measurement ------------------------------------------------------------

namespace
{
/// Per worker, the decode phase may use the boolean window plus the state of
/// one codec. The largest is liblzma, whose dictionary is fixed by the writer
/// (8 MiB at the presets ROOT uses); zstd and zlib are a few hundred KiB.
constexpr int64_t kDecodeScratchBound = 9 * 1024 * 1024;

/// No basket-sized buffer: decoding a value column (not booleans) allocates at
/// most a constant per codec, whatever the basket size. For zstd that is its
/// context plus two 128 KiB blocks; for liblzma, its dictionary. Booleans are
/// reported but not bounded this way: their output cannot be the codec's
/// history, so zstd and LZ4 keep up to one window or chunk of it.
int64_t valueAllocationBound(int compress)
{
  return compress / 100 == 2 ? kDecodeScratchBound : 640 * 1024;
}

struct Phase {
  int64_t start;
  Phase()
  {
    start = gCurrent.load();
    gPeak = start;
    gLargest = 0;
  }
  int64_t peakIncrease() const { return gPeak.load() - start; }
  int64_t retained() const { return gCurrent.load() - start; }
};

int64_t rssAnonKiB()
{
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.starts_with("RssAnon:")) {
      return std::stoll(line.substr(8));
    }
  }
  return -1;
}

std::string mib(int64_t bytes)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f MiB", bytes / 1048576.);
  return buf;
}

struct SharedBuffer {
  uint8_t* data = nullptr;
  size_t size = 0;
  explicit SharedBuffer(size_t n) : size(std::max<size_t>(n, 1))
  {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      throw std::bad_alloc();
    }
    data = (uint8_t*)p;
  }
  ~SharedBuffer() { munmap(data, size); }
};
} // namespace

int main(int argc, char** argv)
{
  Options options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      options.threads = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--pread") == 0) {
      options.localIO = Options::LocalIO::PRead;
    } else {
      files.push_back(argv[i]);
    }
  }
  if (files.empty()) {
    std::fprintf(stderr, "usage: %s [--threads N] [--pread] <file>...\n", argv[0]);
    return 2;
  }
  int failures = 0;
  std::printf("%-18s %-6s %11s %10s %10s %12s %13s %13s %15s %10s\n", "file", "DF", "decoded", "metadata", "offsets",
              "decode peak", "values alloc", "bools alloc", "largest basket", "RssAnon +");
  for (auto& path : files) {
    try {
      auto source = std::shared_ptr<ByteSource>(ByteSource::open(path, options));
      RootFile file{source};
      for (auto& df : file.dataFrames()) {
        auto rssBefore = rssAnonKiB();
        Phase metadata;
        std::vector<TreeInfo> trees;
        for (auto& key : file.keysOf(df)) {
          if (key.className == "TTree") {
            trees.push_back(file.readTree(key));
          }
        }
        auto metadataBytes = metadata.peakIncrease();

        Phase offsets;
        BranchDecoder decoder{*source, options};
        std::vector<std::vector<ColumnOp>> plans;
        for (auto& tree : trees) {
          int64_t rows;
          plans.push_back(planColumns(tree, datasetSchema(physicalSchema(tree)), decoder, arrow::system_memory_pool(), rows));
        }
        auto offsetsBytes = offsets.peakIncrease();

        int64_t decodePeak = 0, valuesAlloc = 0, boolsAlloc = 0, decoded = 0;
        uint32_t largestBasket = 0;
        for (auto& plan : plans) {
          for (auto& op : plan) {
            for (auto& b : op.branch->baskets) {
              largestBasket = std::max(largestBasket, b.nbytes);
            }
            SharedBuffer out{op.byteSize};
            Phase decode;
            executeColumnOp(decoder, op, out.data);
            decodePeak = std::max(decodePeak, decode.peakIncrease());
            auto& largest = op.kind == ColumnOp::Kind::Booleans ? boolsAlloc : valuesAlloc;
            largest = std::max(largest, gLargest.load());
            decoded += op.byteSize;
          }
        }
        // In pread mode, the compressed input of each basket in flight is private too.
        bool pread = options.localIO == Options::LocalIO::PRead;
        int64_t inputAllowance = pread ? (int64_t)options.threads * largestBasket : 0;
        auto rssDelta = (rssAnonKiB() - rssBefore) * 1024;
        std::string verdict = "ok";
        if (decodePeak > options.threads * kDecodeScratchBound + inputAllowance) {
          verdict = "FAIL: decode peak > " + mib(options.threads * kDecodeScratchBound + inputAllowance);
        } else if (!pread && valuesAlloc > valueAllocationBound(file.header().compress)) {
          verdict = "FAIL: value column allocation > " + mib(valueAllocationBound(file.header().compress));
        }
        std::printf("%-18s %-6s %11s %10s %10s %12s %13s %13s %15s %10s %s\n", path.substr(path.rfind('/') + 1).c_str(), df.c_str(),
                    mib(decoded).c_str(), mib(metadataBytes).c_str(), mib(offsetsBytes).c_str(), mib(decodePeak).c_str(),
                    mib(valuesAlloc).c_str(), mib(boolsAlloc).c_str(), mib(largestBasket).c_str(), mib(rssDelta).c_str(), verdict.c_str());
        failures += verdict != "ok";
      }
    } catch (FormatError const& e) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
      ++failures;
    }
  }
  return failures ? 1 : 0;
}
