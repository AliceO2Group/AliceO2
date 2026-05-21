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

/// \file benchmark_ShmemVsMemfd.cxx
/// \brief Head-to-head benchmark: FairMQ shmem transport vs memfd+UDS fd passing
///
/// Self-contained single-file benchmark using fork() for sender/receiver.
/// Approach A: FairMQ shmem push/pull channel with per-message allocation
/// Approach B: memfd (Linux) or shm_open (macOS) + bump allocator + UDS SCM_RIGHTS

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/memfd.h>
#endif

#include <fairmq/Channel.h>
#include <fairmq/Message.h>
#include <fairmq/Parts.h>
#include <fairmq/ProgOptions.h>
#include <fairmq/TransportFactory.h>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
static constexpr int MAX_MESSAGES = 256;
static constexpr int N_ITERATIONS = 1000;
static constexpr size_t ALIGNMENT = 64;

struct Scenario {
  const char* name;
  std::vector<size_t> sizes;
};

// Scenario 1: many small-to-medium messages (realistic TPC-like mix)
static Scenario makeManySmallScenario()
{
  std::vector<size_t> sizes;
  for (int i = 0; i < 50; ++i) {
    sizes.push_back(4 * 1024);
  }
  for (int i = 0; i < 30; ++i) {
    sizes.push_back(64 * 1024);
  }
  for (int i = 0; i < 15; ++i) {
    sizes.push_back(256 * 1024);
  }
  for (int i = 0; i < 5; ++i) {
    sizes.push_back(1024 * 1024);
  }
  return {"100 messages (50x4KB + 30x64KB + 15x256KB + 5x1MB)", std::move(sizes)};
}

// Scenario 2: few large messages
static Scenario makeFewLargeScenario()
{
  std::vector<size_t> sizes;
  for (int i = 0; i < 5; ++i) {
    sizes.push_back(16 * 1024 * 1024); // 5x16MB = 80MB total
  }
  return {"5 messages (5x16MB)", std::move(sizes)};
}

static size_t totalPayloadSize(const std::vector<size_t>& sizes)
{
  return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
}

static size_t alignUp(size_t v, size_t align)
{
  return (v + align - 1) & ~(align - 1);
}

// Fill buffer with a pattern that depends on both iteration and message index,
// so swapped or misrouted messages are detected.
static void fillPattern(void* buf, size_t size, uint8_t iterSeed, int msgIndex)
{
  auto* p = static_cast<uint8_t*>(buf);
  uint8_t base = static_cast<uint8_t>(iterSeed ^ (msgIndex * 37));
  for (size_t i = 0; i < size; ++i) {
    p[i] = static_cast<uint8_t>(base + (i & 0xFF));
  }
}

static bool verifyPattern(const void* buf, size_t size, uint8_t iterSeed, int msgIndex)
{
  auto* p = static_cast<const uint8_t*>(buf);
  uint8_t base = static_cast<uint8_t>(iterSeed ^ (msgIndex * 37));
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != static_cast<uint8_t>(base + (i & 0xFF))) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Timing results communicated from child to parent via pipe
// ---------------------------------------------------------------------------
struct TimingResult {
  double totalMs;
};

struct MemfdReceiverTiming {
  double recvMs;
  double mmapMs;
  double verifyMs;
  double unmapMs;
};

using Clock = std::chrono::high_resolution_clock;

static double msElapsed(Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ---------------------------------------------------------------------------
// Helper: create anonymous shared memory fd (portable)
// ---------------------------------------------------------------------------

static int createAnonymousShmFd(size_t size)
{
#ifdef __linux__
  int fd = memfd_create("benchmark_region", MFD_CLOEXEC);
  if (fd < 0) {
    perror("memfd_create");
    return -1;
  }
#else
  // macOS fallback: shm_open + shm_unlink for an anonymous-like fd
  // shm_open names must be short (max 31 chars on macOS including the leading /)
  static int shmCounter = 0;
  char name[32];
  snprintf(name, sizeof(name), "/bm_%d_%d", getpid(), shmCounter++);
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    perror("shm_open");
    return -1;
  }
  shm_unlink(name); // unlink immediately so it's anonymous
#endif
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    perror("ftruncate");
    close(fd);
    return -1;
  }
  return fd;
}

// ---------------------------------------------------------------------------
// Approach A: FairMQ shmem push/pull
// ---------------------------------------------------------------------------
struct ApproachAResult {
  double allocFillMs;
  double sendMs;
  double receiveMs;
};

static ApproachAResult benchmarkFairMQShmem(const std::vector<size_t>& sizes)
{
  // Use a unique IPC path and session to avoid collisions
  std::string ipcPath = "ipc:///tmp/benchmark_fairmq_" + std::to_string(getpid());

  // Pipe for child to send timing back to parent
  int timePipe[2];
  if (pipe(timePipe) != 0) {
    perror("pipe");
    exit(1);
  }

  // Sync pipe: parent writes a byte after binding, child reads before connecting
  int syncPipe[2];
  if (pipe(syncPipe) != 0) {
    perror("pipe");
    exit(1);
  }

  // Ack pipe: child writes after receiving each batch, parent reads before sending next
  int ackPipe[2];
  if (pipe(ackPipe) != 0) {
    perror("pipe");
    exit(1);
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    exit(1);
  }

  if (pid == 0) {
    // --- Child: receiver (pull) ---
    close(timePipe[0]); // close read end
    close(syncPipe[1]); // close write end
    close(ackPipe[0]);  // close read end of ack pipe

    // Wait for parent to bind
    char syncByte;
    if (read(syncPipe[0], &syncByte, 1) != 1) {
      _exit(1);
    }
    close(syncPipe[0]);

    size_t session = static_cast<size_t>(getppid()) * 1000 + 1;
    fair::mq::ProgOptions config;
    config.SetProperty<std::string>("session", std::to_string(session));
    config.SetProperty<size_t>("shm-segment-size", size_t{2} << 30); // 2 GB

    auto factory = fair::mq::TransportFactory::CreateTransportFactory("shmem", "bench_recv", &config);
    fair::mq::Channel channel("benchmark", "pull", factory);
    channel.Connect(ipcPath);
    channel.Validate();

    double totalReceiveMs = 0.0;

    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
      fair::mq::Parts parts;
      auto t0 = Clock::now();
      auto rc = channel.Receive(parts, 30000); // 30s timeout
      auto t1 = Clock::now();

      if (rc < 0) {
        fprintf(stderr, "FairMQ Receive failed: %ld\n", (long)rc);
        _exit(1);
      }

      // Verify data integrity
      for (int i = 0; i < static_cast<int>(parts.Size()); ++i) {
        if (!verifyPattern(parts[i].GetData(), parts[i].GetSize(),
                           static_cast<uint8_t>(iter & 0xFF), i)) {
          fprintf(stderr, "FairMQ: data verification failed at iter=%d msg=%d\n", iter, i);
          _exit(1);
        }
      }
      totalReceiveMs += msElapsed(t0, t1);

      // Ack: signal sender that we've consumed this batch
      char ack = 'A';
      if (write(ackPipe[1], &ack, 1) != 1) {
        perror("write ack");
        _exit(1);
      }
    }

    close(ackPipe[1]);
    TimingResult result{totalReceiveMs};
    if (write(timePipe[1], &result, sizeof(result)) != sizeof(result)) {
      perror("write timing");
    }
    close(timePipe[1]);
    _exit(0);
  }

  // --- Parent: sender (push) ---
  close(timePipe[1]); // close write end
  close(syncPipe[0]); // close read end
  close(ackPipe[1]);  // close write end of ack pipe

  size_t session = static_cast<size_t>(getpid()) * 1000 + 1;
  size_t shmSegSize = size_t{2} << 30; // 2 GB
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));
  config.SetProperty<size_t>("shm-segment-size", shmSegSize);

  auto factory = fair::mq::TransportFactory::CreateTransportFactory("shmem", "bench_send", &config);
  fair::mq::Channel channel("benchmark", "push", factory);
  channel.Bind(ipcPath);
  channel.Validate();

  // Signal child that we've bound
  char syncByte = 'G';
  if (write(syncPipe[1], &syncByte, 1) != 1) {
    perror("write sync");
  }
  close(syncPipe[1]);

  // Give child a moment to connect
  usleep(50000);

  double totalAllocFillMs = 0.0;
  double totalSendMs = 0.0;

  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    fair::mq::Parts parts;

    auto t0 = Clock::now();
    for (int m = 0; m < static_cast<int>(sizes.size()); ++m) {
      auto msg = factory->CreateMessage(sizes[m]);
      fillPattern(msg->GetData(), sizes[m], static_cast<uint8_t>(iter & 0xFF), m);
      parts.AddPart(std::move(msg));
    }
    auto t1 = Clock::now();

    auto rc = channel.Send(parts, 30000);
    auto t2 = Clock::now();

    if (rc < 0) {
      fprintf(stderr, "FairMQ Send failed: %ld\n", (long)rc);
      exit(1);
    }

    totalAllocFillMs += msElapsed(t0, t1);
    totalSendMs += msElapsed(t1, t2);

    // Wait for receiver to consume this batch before sending next
    char ack;
    if (read(ackPipe[0], &ack, 1) != 1) {
      fprintf(stderr, "FairMQ: failed to read ack at iter=%d\n", iter);
      exit(1);
    }
  }

  close(ackPipe[0]);

  // Read child timing
  TimingResult childResult{};
  if (read(timePipe[0], &childResult, sizeof(childResult)) != sizeof(childResult)) {
    perror("read timing");
  }
  close(timePipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FairMQ child exited abnormally\n");
  }

  // Clean up IPC file
  std::string ipcFile = "/tmp/benchmark_fairmq_" + std::to_string(getpid());
  unlink(ipcFile.c_str());

  return ApproachAResult{
    totalAllocFillMs / N_ITERATIONS,
    totalSendMs / N_ITERATIONS,
    childResult.totalMs / N_ITERATIONS};
}

// ---------------------------------------------------------------------------
// Approach B: memfd + bump allocator + UDS fd passing
// ---------------------------------------------------------------------------

// Manifest entry describing one message within the shared region
struct ManifestEntry {
  uint32_t offset;
  uint32_t size;
};

struct Manifest {
  uint32_t count;
  uint32_t totalSize;
  ManifestEntry entries[MAX_MESSAGES];
};

// Send fd + manifest over UDS using SCM_RIGHTS
static bool sendFdAndManifest(int sockFd, int shmFd, const Manifest& manifest)
{
  struct msghdr msg = {};
  struct iovec iov = {};
  iov.iov_base = const_cast<Manifest*>(&manifest);
  iov.iov_len = sizeof(manifest);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  // Ancillary data for SCM_RIGHTS
  union {
    char buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr align;
  } cmsgBuf = {};

  msg.msg_control = cmsgBuf.buf;
  msg.msg_controllen = sizeof(cmsgBuf.buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &shmFd, sizeof(int));

  ssize_t sent = sendmsg(sockFd, &msg, 0);
  return sent >= 0;
}

// Receive fd + manifest from UDS
static bool recvFdAndManifest(int sockFd, int& shmFd, Manifest& manifest)
{
  struct msghdr msg = {};
  struct iovec iov = {};
  iov.iov_base = &manifest;
  iov.iov_len = sizeof(manifest);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  union {
    char buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr align;
  } cmsgBuf = {};

  msg.msg_control = cmsgBuf.buf;
  msg.msg_controllen = sizeof(cmsgBuf.buf);

  ssize_t received = recvmsg(sockFd, &msg, 0);
  if (received < static_cast<ssize_t>(sizeof(manifest))) {
    return false;
  }

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    memcpy(&shmFd, CMSG_DATA(cmsg), sizeof(int));
    return true;
  }
  return false;
}

struct ApproachBResult {
  double memfdCreateMs;   // memfd_create + ftruncate
  double senderMmapMs;    // mmap on sender
  double fillMs;          // fill pattern
  double sendMs;          // sendmsg (fd + manifest)
  double senderUnmapMs;   // munmap + close on sender
  double recvMs;          // recvmsg (fd + manifest)
  double receiverMmapMs;  // mmap on receiver
  double verifyMs;        // verify pattern
  double receiverUnmapMs; // munmap + close on receiver
};

static ApproachBResult benchmarkMemfdUDS(const std::vector<size_t>& sizes)
{
  std::string sockPath = "/tmp/benchmark_memfd_" + std::to_string(getpid()) + ".sock";
  unlink(sockPath.c_str());

  // Pipe for child to send timing back
  int timePipe[2];
  if (pipe(timePipe) != 0) {
    perror("pipe");
    exit(1);
  }

  // Sync pipe: parent writes after listen(), child reads before connect()
  int syncPipe[2];
  if (pipe(syncPipe) != 0) {
    perror("pipe");
    exit(1);
  }

  // Compute total bump region size (with alignment)
  size_t regionSize = 0;
  for (int m = 0; m < static_cast<int>(sizes.size()); ++m) {
    regionSize += alignUp(sizes[m], ALIGNMENT);
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    exit(1);
  }

  if (pid == 0) {
    // --- Child: receiver ---
    close(timePipe[0]);
    close(syncPipe[1]);

    // Wait for parent to listen
    char syncByte;
    if (read(syncPipe[0], &syncByte, 1) != 1) {
      _exit(1);
    }
    close(syncPipe[0]);

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
      perror("socket");
      _exit(1);
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
      perror("connect");
      _exit(1);
    }

    MemfdReceiverTiming timing{};

    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
      Manifest manifest{};
      int shmFd = -1;

      auto t0 = Clock::now();
      if (!recvFdAndManifest(sock, shmFd, manifest)) {
        fprintf(stderr, "memfd: recvFdAndManifest failed at iter=%d\n", iter);
        _exit(1);
      }
      auto t1 = Clock::now();

      int mmapFlags = MAP_SHARED;
#ifdef MAP_POPULATE
      mmapFlags |= MAP_POPULATE;
#endif
      void* region = mmap(nullptr, manifest.totalSize, PROT_READ, mmapFlags, shmFd, 0);
      if (region == MAP_FAILED) {
        perror("mmap receiver");
        _exit(1);
      }
      auto t2 = Clock::now();

      // Verify
      for (uint32_t m = 0; m < manifest.count; ++m) {
        const auto& entry = manifest.entries[m];
        if (!verifyPattern(static_cast<const uint8_t*>(region) + entry.offset,
                           entry.size, static_cast<uint8_t>(iter & 0xFF), static_cast<int>(m))) {
          fprintf(stderr, "memfd: data verification failed at iter=%d msg=%u\n", iter, m);
          _exit(1);
        }
      }
      auto t3 = Clock::now();

      munmap(region, manifest.totalSize);
      close(shmFd);
      auto t4 = Clock::now();

      timing.recvMs += msElapsed(t0, t1);
      timing.mmapMs += msElapsed(t1, t2);
      timing.verifyMs += msElapsed(t2, t3);
      timing.unmapMs += msElapsed(t3, t4);
    }

    close(sock);

    if (write(timePipe[1], &timing, sizeof(timing)) != sizeof(timing)) {
      perror("write timing");
    }
    close(timePipe[1]);
    _exit(0);
  }

  // --- Parent: sender ---
  close(timePipe[1]);
  close(syncPipe[0]);

  int listenSock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listenSock < 0) {
    perror("socket");
    exit(1);
  }

  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(listenSock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("bind");
    exit(1);
  }
  if (listen(listenSock, 1) != 0) {
    perror("listen");
    exit(1);
  }

  // Signal child that we're listening
  char syncByte = 'G';
  if (write(syncPipe[1], &syncByte, 1) != 1) {
    perror("write sync");
  }
  close(syncPipe[1]);

  int connSock = accept(listenSock, nullptr, nullptr);
  if (connSock < 0) {
    perror("accept");
    exit(1);
  }

  double totalMemfdCreateMs = 0.0;
  double totalSenderMmapMs = 0.0;
  double totalFillMs = 0.0;
  double totalSendMs = 0.0;
  double totalSenderUnmapMs = 0.0;

  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    auto t0 = Clock::now();

    // Create anonymous shared memory region
    int shmFd = createAnonymousShmFd(regionSize);
    if (shmFd < 0) {
      exit(1);
    }
    auto t1 = Clock::now();

    int senderMmapFlags = MAP_SHARED;
#ifdef MAP_POPULATE
    senderMmapFlags |= MAP_POPULATE;
#endif
    void* region = mmap(nullptr, regionSize, PROT_READ | PROT_WRITE, senderMmapFlags, shmFd, 0);
    if (region == MAP_FAILED) {
      perror("mmap sender");
      exit(1);
    }
    auto t2 = Clock::now();

    // Bump-allocate and fill
    Manifest manifest{};
    manifest.count = static_cast<uint32_t>(sizes.size());
    manifest.totalSize = static_cast<uint32_t>(regionSize);
    size_t offset = 0;
    for (int m = 0; m < static_cast<int>(sizes.size()); ++m) {
      manifest.entries[m].offset = static_cast<uint32_t>(offset);
      manifest.entries[m].size = static_cast<uint32_t>(sizes[m]);
      fillPattern(static_cast<uint8_t*>(region) + offset, sizes[m],
                  static_cast<uint8_t>(iter & 0xFF), m);
      offset += alignUp(sizes[m], ALIGNMENT);
    }
    auto t3 = Clock::now();

    // Unmap before sending — pages remain in the shm/memfd object
    munmap(region, regionSize);
    auto t4 = Clock::now();

    // Send fd + manifest
    if (!sendFdAndManifest(connSock, shmFd, manifest)) {
      fprintf(stderr, "memfd: sendFdAndManifest failed at iter=%d\n", iter);
      exit(1);
    }
    auto t5 = Clock::now();

    close(shmFd);
    auto t6 = Clock::now();

    totalMemfdCreateMs += msElapsed(t0, t1);
    totalSenderMmapMs += msElapsed(t1, t2);
    totalFillMs += msElapsed(t2, t3);
    totalSenderUnmapMs += msElapsed(t3, t4) + msElapsed(t5, t6); // munmap + close
    totalSendMs += msElapsed(t4, t5);
  }

  close(connSock);
  close(listenSock);
  unlink(sockPath.c_str());

  // Read child timing
  MemfdReceiverTiming childTiming{};
  if (read(timePipe[0], &childTiming, sizeof(childTiming)) != sizeof(childTiming)) {
    perror("read timing");
  }
  close(timePipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "memfd child exited abnormally\n");
  }

  return ApproachBResult{
    totalMemfdCreateMs / N_ITERATIONS,
    totalSenderMmapMs / N_ITERATIONS,
    totalFillMs / N_ITERATIONS,
    totalSendMs / N_ITERATIONS,
    totalSenderUnmapMs / N_ITERATIONS,
    childTiming.recvMs / N_ITERATIONS,
    childTiming.mmapMs / N_ITERATIONS,
    childTiming.verifyMs / N_ITERATIONS,
    childTiming.unmapMs / N_ITERATIONS};
}

// ---------------------------------------------------------------------------
// Approach C: slab-based memfd with reuse + madvise
// ---------------------------------------------------------------------------
// Pre-create a few large slabs (memfds). Bump-allocate TFs into them.
// When a slab fills, move to the next. When receiver finishes a slab,
// it sends an ack and calls madvise(MADV_DONTNEED). Sender reuses the slab.

static constexpr int N_SLABS = 4;
static constexpr size_t SLAB_SIZE = 128 * 1024 * 1024; // 128MB per slab

// Slab manifest: which slab, where in it, and per-message layout
struct SlabManifest {
  int32_t slabIndex;   // which slab this TF is in
  int32_t tfIndex;     // TF iteration number (for verification)
  uint32_t count;      // number of messages
  uint32_t baseOffset; // offset within slab where this TF starts
  uint32_t totalSize;  // total bytes used by this TF
  bool lastInSlab;     // true if this is the last TF that fits in the slab
  ManifestEntry entries[MAX_MESSAGES];
};

// Send/recv for slab manifest (plain data, no fd passing)
static bool sendSlabManifest(int sockFd, const SlabManifest& manifest)
{
  ssize_t sent = send(sockFd, &manifest, sizeof(manifest), 0);
  return sent == sizeof(manifest);
}

static bool recvSlabManifest(int sockFd, SlabManifest& manifest)
{
  size_t remaining = sizeof(manifest);
  char* buf = reinterpret_cast<char*>(&manifest);
  while (remaining > 0) {
    ssize_t n = recv(sockFd, buf, remaining, 0);
    if (n <= 0) {
      return false;
    }
    buf += n;
    remaining -= n;
  }
  return true;
}

// Ack message: receiver tells sender which slab is fully consumed
struct SlabAck {
  int32_t slabIndex; // -1 means just a per-TF ack, no slab release
};

static bool sendSlabAck(int sockFd, const SlabAck& ack)
{
  return send(sockFd, &ack, sizeof(ack), 0) == sizeof(ack);
}

static bool recvSlabAck(int sockFd, SlabAck& ack)
{
  return recv(sockFd, &ack, sizeof(ack), MSG_WAITALL) == sizeof(ack);
}

struct ApproachCResult {
  double fillMs;
  double sendManifestMs;
  double recvManifestMs;
  double verifyMs;
  double madviseMs;
};

struct SlabReceiverTiming {
  double recvManifestMs;
  double verifyMs;
  double madviseMs;
};

static ApproachCResult benchmarkSlabMemfd(const std::vector<size_t>& sizes)
{
  std::string sockPath = "/tmp/benchmark_slab_" + std::to_string(getpid()) + ".sock";
  unlink(sockPath.c_str());

  // Compute per-TF size
  size_t tfSize = 0;
  for (size_t s : sizes) {
    tfSize += alignUp(s, ALIGNMENT);
  }

  int timePipe[2];
  if (pipe(timePipe) != 0) {
    perror("pipe");
    exit(1);
  }

  int syncPipe[2];
  if (pipe(syncPipe) != 0) {
    perror("pipe");
    exit(1);
  }

  // Create slabs before fork so both processes inherit the fds
  int slabFds[N_SLABS];
  for (int i = 0; i < N_SLABS; ++i) {
    slabFds[i] = createAnonymousShmFd(SLAB_SIZE);
    if (slabFds[i] < 0) {
      fprintf(stderr, "Failed to create slab %d\n", i);
      exit(1);
    }
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    exit(1);
  }

  if (pid == 0) {
    // --- Child: receiver ---
    close(timePipe[0]);
    close(syncPipe[1]);

    char syncByte;
    if (read(syncPipe[0], &syncByte, 1) != 1) {
      _exit(1);
    }
    close(syncPipe[0]);

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
      perror("socket");
      _exit(1);
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
      perror("connect");
      _exit(1);
    }

    // mmap all slabs PROT_READ
    void* slabMaps[N_SLABS];
    for (int i = 0; i < N_SLABS; ++i) {
      slabMaps[i] = mmap(nullptr, SLAB_SIZE, PROT_READ, MAP_SHARED, slabFds[i], 0);
      if (slabMaps[i] == MAP_FAILED) {
        perror("mmap slab receiver");
        _exit(1);
      }
    }

    SlabReceiverTiming timing{};

    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
      SlabManifest manifest{};

      auto t0 = Clock::now();
      if (!recvSlabManifest(sock, manifest)) {
        fprintf(stderr, "slab: recvSlabManifest failed at iter=%d\n", iter);
        _exit(1);
      }
      auto t1 = Clock::now();

      // Verify
      auto* base = static_cast<const uint8_t*>(slabMaps[manifest.slabIndex]);
      for (uint32_t m = 0; m < manifest.count; ++m) {
        const auto& entry = manifest.entries[m];
        if (!verifyPattern(base + manifest.baseOffset + entry.offset,
                           entry.size, static_cast<uint8_t>(manifest.tfIndex & 0xFF),
                           static_cast<int>(m))) {
          fprintf(stderr, "slab: data verification failed at iter=%d msg=%u slab=%d\n",
                  iter, m, manifest.slabIndex);
          _exit(1);
        }
      }
      auto t2 = Clock::now();

      double madvMs = 0.0;
      SlabAck ack{-1};
      if (manifest.lastInSlab) {
        // Release entire slab's pages
        auto tm0 = Clock::now();
#ifdef MADV_DONTNEED
        madvise(slabMaps[manifest.slabIndex], SLAB_SIZE, MADV_DONTNEED);
#endif
        auto tm1 = Clock::now();
        madvMs = msElapsed(tm0, tm1);
        ack.slabIndex = manifest.slabIndex;
      }

      if (!sendSlabAck(sock, ack)) {
        perror("sendSlabAck");
        _exit(1);
      }

      timing.recvManifestMs += msElapsed(t0, t1);
      timing.verifyMs += msElapsed(t1, t2);
      timing.madviseMs += madvMs;
    }

    // Clean up
    for (int i = 0; i < N_SLABS; ++i) {
      munmap(slabMaps[i], SLAB_SIZE);
      close(slabFds[i]);
    }
    close(sock);

    if (write(timePipe[1], &timing, sizeof(timing)) != sizeof(timing)) {
      perror("write timing");
    }
    close(timePipe[1]);
    _exit(0);
  }

  // --- Parent: sender ---
  close(timePipe[1]);
  close(syncPipe[0]);

  int listenSock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listenSock < 0) {
    perror("socket");
    exit(1);
  }

  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(listenSock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("bind");
    exit(1);
  }
  if (listen(listenSock, 1) != 0) {
    perror("listen");
    exit(1);
  }

  char syncByte = 'G';
  if (write(syncPipe[1], &syncByte, 1) != 1) {
    perror("write sync");
  }
  close(syncPipe[1]);

  int connSock = accept(listenSock, nullptr, nullptr);
  if (connSock < 0) {
    perror("accept");
    exit(1);
  }

  // mmap all slabs PROT_READ|PROT_WRITE
  void* slabMaps[N_SLABS];
  for (int i = 0; i < N_SLABS; ++i) {
    int flags = MAP_SHARED;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    slabMaps[i] = mmap(nullptr, SLAB_SIZE, PROT_READ | PROT_WRITE, flags, slabFds[i], 0);
    if (slabMaps[i] == MAP_FAILED) {
      perror("mmap slab sender");
      exit(1);
    }
  }

  // Track which slabs are available (not being read by receiver)
  bool slabAvailable[N_SLABS];
  for (int i = 0; i < N_SLABS; ++i) {
    slabAvailable[i] = true;
  }

  double totalFillMs = 0.0;
  double totalSendManifestMs = 0.0;

  int currentSlab = 0;
  size_t slabOffset = 0;
  slabAvailable[0] = false;

  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    // Check if current TF fits in current slab
    if (slabOffset + tfSize > SLAB_SIZE) {
      // Move to next slab
      currentSlab = (currentSlab + 1) % N_SLABS;
      slabOffset = 0;

      // Wait until this slab is available (drain acks until it is)
      while (!slabAvailable[currentSlab]) {
        SlabAck ack{};
        if (!recvSlabAck(connSock, ack)) {
          fprintf(stderr, "slab: recvSlabAck failed waiting for slab %d\n", currentSlab);
          exit(1);
        }
        if (ack.slabIndex >= 0) {
          slabAvailable[ack.slabIndex] = true;
        }
      }
      slabAvailable[currentSlab] = false;
    }

    auto t0 = Clock::now();

    // Bump-allocate and fill in current slab
    auto* base = static_cast<uint8_t*>(slabMaps[currentSlab]);
    SlabManifest manifest{};
    manifest.slabIndex = currentSlab;
    manifest.tfIndex = iter;
    manifest.count = static_cast<uint32_t>(sizes.size());
    manifest.baseOffset = static_cast<uint32_t>(slabOffset);
    manifest.totalSize = static_cast<uint32_t>(tfSize);
    manifest.lastInSlab = (slabOffset + tfSize + tfSize > SLAB_SIZE);

    size_t localOffset = 0;
    for (int m = 0; m < static_cast<int>(sizes.size()); ++m) {
      manifest.entries[m].offset = static_cast<uint32_t>(localOffset);
      manifest.entries[m].size = static_cast<uint32_t>(sizes[m]);
      fillPattern(base + slabOffset + localOffset, sizes[m],
                  static_cast<uint8_t>(iter & 0xFF), m);
      localOffset += alignUp(sizes[m], ALIGNMENT);
    }
    auto t1 = Clock::now();

    if (!sendSlabManifest(connSock, manifest)) {
      fprintf(stderr, "slab: sendSlabManifest failed at iter=%d\n", iter);
      exit(1);
    }
    auto t2 = Clock::now();

    slabOffset += tfSize;

    totalFillMs += msElapsed(t0, t1);
    totalSendManifestMs += msElapsed(t1, t2);

    // Read one ack per TF to stay in sync
    SlabAck ack{};
    if (!recvSlabAck(connSock, ack)) {
      fprintf(stderr, "slab: recvSlabAck failed at iter=%d\n", iter);
      exit(1);
    }
    if (ack.slabIndex >= 0) {
      slabAvailable[ack.slabIndex] = true;
    }
  }

  // Clean up
  for (int i = 0; i < N_SLABS; ++i) {
    munmap(slabMaps[i], SLAB_SIZE);
    close(slabFds[i]);
  }
  close(connSock);
  close(listenSock);
  unlink(sockPath.c_str());

  // Read child timing
  SlabReceiverTiming childTiming{};
  if (read(timePipe[0], &childTiming, sizeof(childTiming)) != sizeof(childTiming)) {
    perror("read timing");
  }
  close(timePipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "slab child exited abnormally\n");
  }

  return ApproachCResult{
    totalFillMs / N_ITERATIONS,
    totalSendManifestMs / N_ITERATIONS,
    childTiming.recvManifestMs / N_ITERATIONS,
    childTiming.verifyMs / N_ITERATIONS,
    childTiming.madviseMs / N_ITERATIONS};
}

// ---------------------------------------------------------------------------
// Run one scenario and print results
// ---------------------------------------------------------------------------
static void runScenario(const Scenario& scenario)
{
  const auto& sizes = scenario.sizes;
  int nMessages = static_cast<int>(sizes.size());
  size_t totalBytes = totalPayloadSize(sizes);
  double totalMB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);

  printf("--------------------------------------------------------------\n");
  printf("Scenario: %s\n", scenario.name);
  printf("  Total payload:   %.2f MB per TF\n", totalMB);
  printf("  Iterations:      %d\n\n", N_ITERATIONS);

  printf("Running FairMQ shmem benchmark...\n");
  auto resultA = benchmarkFairMQShmem(sizes);

  printf("Running memfd+UDS benchmark...\n");
  auto resultB = benchmarkMemfdUDS(sizes);

  printf("Running slab memfd benchmark...\n");
  auto resultC = benchmarkSlabMemfd(sizes);

  double totalA = resultA.allocFillMs + resultA.sendMs + resultA.receiveMs;
  double throughputA = totalMB / (totalA / 1000.0);

  double senderB = resultB.memfdCreateMs + resultB.senderMmapMs + resultB.fillMs + resultB.sendMs + resultB.senderUnmapMs;
  double receiverB = resultB.recvMs + resultB.receiverMmapMs + resultB.verifyMs + resultB.receiverUnmapMs;
  double totalB = senderB + receiverB;
  double throughputB = totalMB / (totalB / 1000.0);

  printf("\n=== FairMQ shmem (%d iterations, %d messages/TF) ===\n",
         N_ITERATIONS, nMessages);
  printf("  Alloc+Fill:  %.2f ms/TF\n", resultA.allocFillMs);
  printf("  Send:        %.2f ms/TF\n", resultA.sendMs);
  printf("  Receive:     %.2f ms/TF\n", resultA.receiveMs);
  printf("  Total:       %.2f ms/TF\n", totalA);
  printf("  Throughput:  %.2f GB/s\n", throughputA / 1024.0);

  printf("\n=== memfd + bump + UDS (%d iterations, %d messages/TF) ===\n",
         N_ITERATIONS, nMessages);
  printf("  Sender breakdown:\n");
  printf("    memfd_create: %.2f ms/TF\n", resultB.memfdCreateMs);
  printf("    mmap:         %.2f ms/TF\n", resultB.senderMmapMs);
  printf("    fill:         %.2f ms/TF\n", resultB.fillMs);
  printf("    sendmsg:      %.2f ms/TF\n", resultB.sendMs);
  printf("    munmap+close: %.2f ms/TF\n", resultB.senderUnmapMs);
  printf("    subtotal:     %.2f ms/TF\n", senderB);
  printf("  Receiver breakdown:\n");
  printf("    recvmsg:      %.2f ms/TF\n", resultB.recvMs);
  printf("    mmap:         %.2f ms/TF\n", resultB.receiverMmapMs);
  printf("    verify:       %.2f ms/TF\n", resultB.verifyMs);
  printf("    munmap+close: %.2f ms/TF\n", resultB.receiverUnmapMs);
  printf("    subtotal:     %.2f ms/TF\n", receiverB);
  printf("  Total:       %.2f ms/TF\n", totalB);
  printf("  Throughput:  %.2f GB/s\n", throughputB / 1024.0);

  printf("\nSpeedup (memfd vs FairMQ): %.1fx\n", totalA / totalB);

  double senderC = resultC.fillMs + resultC.sendManifestMs;
  double receiverC = resultC.recvManifestMs + resultC.verifyMs + resultC.madviseMs;
  double totalC = senderC + receiverC;
  double throughputC = totalMB / (totalC / 1000.0);

  printf("\n=== slab memfd + madvise (%d iterations, %d messages/TF, %d slabs x %zuMB) ===\n",
         N_ITERATIONS, nMessages, N_SLABS, SLAB_SIZE / (1024 * 1024));
  printf("  Sender breakdown:\n");
  printf("    fill:         %.2f ms/TF\n", resultC.fillMs);
  printf("    send manifest:%.2f ms/TF\n", resultC.sendManifestMs);
  printf("    subtotal:     %.2f ms/TF\n", senderC);
  printf("  Receiver breakdown:\n");
  printf("    recv manifest:%.2f ms/TF\n", resultC.recvManifestMs);
  printf("    verify:       %.2f ms/TF\n", resultC.verifyMs);
  printf("    madvise:      %.2f ms/TF\n", resultC.madviseMs);
  printf("    subtotal:     %.2f ms/TF\n", receiverC);
  printf("  Total:       %.2f ms/TF\n", totalC);
  printf("  Throughput:  %.2f GB/s\n", throughputC / 1024.0);

  printf("\nSpeedup (slab vs FairMQ): %.1fx\n\n", totalA / totalC);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
  printf("Benchmark: FairMQ shmem vs memfd+UDS\n\n");

  auto scenario1 = makeManySmallScenario();
  auto scenario2 = makeFewLargeScenario();

  runScenario(scenario1);
  runScenario(scenario2);

  return 0;
}
