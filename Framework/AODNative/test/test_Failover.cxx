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

#include "FailoverByteSource.h"
#include "AODNative/Errors.h"
#include <catch_amalgamated.hpp>
#include <atomic>
#include <map>
#include <thread>

using namespace o2::aod::native;

namespace
{
// A replica whose byte i is i % 251. Its behaviour is shared between the
// instances opened for the same URL, so that a reopen sees the same state.
struct Behaviour {
  size_t size = 1000;
  bool opens = true;
  std::atomic<int> failingReads = 0; // the next reads which fail with IoError
  std::atomic<int> opened = 0;
  std::atomic<int> reads = 0;
};

struct FakeSource : ByteSource {
  FakeSource(std::string u, Behaviour& b) : name(std::move(u)), behaviour(b) {}
  uint64_t size() const override { return behaviour.size; }
  void readInto(uint64_t off, size_t len, void* dst) override
  {
    ReadRequest r{off, len, dst};
    readvInto({&r, 1});
  }
  void readvInto(std::span<ReadRequest> requests) override
  {
    ++behaviour.reads;
    int failing = behaviour.failingReads.load();
    while (failing > 0 && !behaviour.failingReads.compare_exchange_weak(failing, failing - 1)) {
    }
    if (failing > 0) {
      throw IoError(name, "Operation expired");
    }
    for (auto& r : requests) {
      for (size_t i = 0; i < r.length; ++i) {
        ((uint8_t*)r.dst)[i] = (r.offset + i) % 251;
      }
      bytes += r.length;
    }
    ++calls;
  }
  bool isRemote() const override { return true; }
  uint64_t bytesRead() const override { return bytes; }
  uint64_t readCalls() const override { return calls; }
  std::string url() const override { return name; }
  std::string name;
  Behaviour& behaviour;
  std::atomic<uint64_t> bytes = 0;
  std::atomic<uint64_t> calls = 0;
};

struct Setup {
  std::map<std::string, Behaviour> replicas;
  std::vector<std::string> log;
  Options options;
  Setup() { options.retryDelay = 0; }

  std::unique_ptr<FailoverByteSource> open()
  {
    std::vector<Replica> list;
    for (auto& [url, _] : replicas) {
      list.push_back({url, "SE-" + url});
    }
    auto opener = [this](std::string const& url, Options const&) -> std::unique_ptr<ByteSource> {
      auto& b = replicas.at(url);
      ++b.opened;
      if (!b.opens) {
        throw IoError(url, "no such file");
      }
      return std::make_unique<FakeSource>(url, b);
    };
    return std::make_unique<FailoverByteSource>(list, options, opener, [this](std::string const& m) { log.push_back(m); });
  }
};

bool readsCorrectly(ByteSource& s, uint64_t off = 100, size_t len = 50)
{
  std::vector<uint8_t> buf(len);
  s.readInto(off, len, buf.data());
  for (size_t i = 0; i < len; ++i) {
    if (buf[i] != (off + i) % 251) {
      return false;
    }
  }
  return true;
}
} // namespace

TEST_CASE("FailoverByteSource reads the first replica")
{
  Setup s;
  s.replicas["a"];
  s.replicas["b"];
  auto f = s.open();
  REQUIRE(readsCorrectly(*f));
  CHECK(f->replicaLabel() == "SE-a");
  CHECK(s.replicas["b"].opened == 0);
  CHECK(f->bytesRead() == 50);
  CHECK(f->readCalls() == 1);
  CHECK(s.log.empty());
}

TEST_CASE("FailoverByteSource skips replicas which do not open")
{
  Setup s;
  s.replicas["a"].opens = false;
  s.replicas["b"];
  auto f = s.open();
  CHECK(f->replicaLabel() == "SE-b");
  REQUIRE(readsCorrectly(*f));
  CHECK(s.log.size() == 1);
}

TEST_CASE("FailoverByteSource throws when no replica opens")
{
  Setup s;
  s.replicas["a"].opens = false;
  s.replicas["b"].opens = false;
  CHECK_THROWS_AS(s.open(), IoError);
}

TEST_CASE("FailoverByteSource retries a transient read error on the same replica")
{
  Setup s;
  s.replicas["a"].failingReads = 1;
  s.replicas["b"];
  auto f = s.open();
  REQUIRE(readsCorrectly(*f));
  CHECK(f->replicaLabel() == "SE-a");
  CHECK(s.replicas["a"].opened == 2); // reopened for the retry
  CHECK(s.replicas["b"].opened == 0);
  REQUIRE(s.log.size() == 1);
  CHECK(s.log[0].find("retry 1 of 2") != std::string::npos);
}

TEST_CASE("FailoverByteSource moves to the next replica after the retries")
{
  Setup s;
  s.replicas["a"].failingReads = 100;
  s.replicas["b"];
  auto f = s.open();
  REQUIRE(readsCorrectly(*f));
  CHECK(f->replicaLabel() == "SE-b");
  CHECK(s.replicas["a"].reads == 3); // first attempt + 2 retries
  CHECK(s.log.back().find("Moving to replica on SE-b") != std::string::npos);
  // Later reads stay on the new replica.
  REQUIRE(readsCorrectly(*f, 900, 100));
  CHECK(s.replicas["a"].reads == 3);
}

TEST_CASE("FailoverByteSource does not use a replica of a different size")
{
  Setup s;
  s.replicas["a"].failingReads = 100;
  s.replicas["b"].size = 999;
  s.replicas["c"];
  auto f = s.open();
  REQUIRE(readsCorrectly(*f));
  CHECK(f->replicaLabel() == "SE-c");
}

TEST_CASE("FailoverByteSource throws when every replica fails")
{
  Setup s;
  s.replicas["a"].failingReads = 100;
  s.replicas["b"].failingReads = 100;
  auto f = s.open();
  std::vector<uint8_t> buf(10);
  try {
    f->readInto(0, 10, buf.data());
    FAIL("no exception");
  } catch (IoError const& e) {
    CHECK(std::string(e.what()).find("no replica left") != std::string::npos);
  }
  CHECK(s.replicas["a"].reads == 3);
  CHECK(s.replicas["b"].reads == 3);
}

TEST_CASE("FailoverByteSource counts the bytes of retired replicas")
{
  Setup s;
  s.replicas["a"];
  s.replicas["b"];
  auto f = s.open();
  REQUIRE(readsCorrectly(*f, 0, 100));
  s.replicas["a"].failingReads = 100;
  REQUIRE(readsCorrectly(*f, 0, 30));
  CHECK(f->bytesRead() == 130);
  CHECK(f->readCalls() == 2);
}

TEST_CASE("FailoverByteSource with concurrent readers")
{
  Setup s;
  s.replicas["a"].failingReads = 100;
  s.replicas["b"];
  auto f = s.open();
  std::vector<std::thread> threads;
  std::atomic<int> good = 0;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 20; ++i) {
        good += readsCorrectly(*f, 10 * t + i, 40);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  CHECK(good == 80);
  CHECK(f->replicaLabel() == "SE-b");
}
