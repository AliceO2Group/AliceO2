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

/// \file dataSamplingSharedPayload.cxx
/// \brief Stress test of the Dispatcher forwarding payloads by shallow copy.
///
/// The producer mimics STFSender: payloads live in an unmanaged shared memory region and are
/// handed over to DPL without a copy. As soon as FairMQ reports a buffer as released, the producer
/// poisons it and puts it back into its free list, to be reused for the next timeframes.
/// The same messages are read concurrently by a regular consumer and, via the Dispatcher,
/// by two "QC-like" consumers (two policies sampling the same data, i.e. two shallow copies per part).
/// All consumers verify the payload twice, with a pause in between, while holding the message.
/// Any premature release of a shared buffer would make the verification fail.
/// The sampled consumers also check that they receive the region buffers themselves (no deep copy).
///
/// How to run it:
///
///   # build (from the O2 build directory, with the O2 environment loaded)
///   ninja stage/bin/o2-datasampling-datasampling-shared-payload
///
///   # default run: 500 timeframes x 4 parts x 1 MB, 32 buffers in the region
///   o2-datasampling-datasampling-shared-payload -b --run --shm-segment-size 1000000000
///
///   # make the direct consumer read the buffers at the same time as the Dispatcher and the sampled consumers
///   # (by default DPL forwards the data to the Dispatcher only after the direct consumer has processed it)
///   o2-datasampling-datasampling-shared-payload -b --run --shm-segment-size 1000000000 --early-forward-policy always
///
///   # heavier run: small region, so that buffers are released, poisoned and reused all the time
///   o2-datasampling-datasampling-shared-payload -b --run --shm-segment-size 1000000000 --early-forward-policy always \
///     --timeframes 3000 --parts 8 --block-size 65536 --blocks 24
///
/// The workflow exits with 0 on success. Each consumer prints a summary at the end of stream, e.g.
///   Checked 2000 parts, 0 errors, 2000 in the producer's region, 0 copied, 850 seen while still shared with another reader
/// It exits with a non-zero code if a payload fails the verification ("payload has been poisoned", "corrupted word"),
/// if a consumer receives no data, or if sampled payloads were deep copies ("... sampled payloads were deep copies").
/// The latter check can be disabled with --allow-deep-copy, e.g. to run the test against a Dispatcher which copies.
/// Only the shared memory transport (the DPL default) is supported.
///
/// Generated with Claude

#include "DataSampling/DataSampling.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ChannelConfigurationPolicy.h"

#include <vector>

using namespace o2::framework;
using namespace o2::utilities;

void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"timeframes", VariantType::Int, 500, {"number of timeframes to produce"}});
  workflowOptions.push_back(ConfigParamSpec{"parts", VariantType::Int, 4, {"number of parts (subspecs) per timeframe"}});
  workflowOptions.push_back(ConfigParamSpec{"block-size", VariantType::Int, 1 << 20, {"size of each payload in bytes"}});
  workflowOptions.push_back(ConfigParamSpec{"blocks", VariantType::Int, 32, {"number of payload buffers in the unmanaged region"}});
  workflowOptions.push_back(ConfigParamSpec{"allow-deep-copy", VariantType::Bool, false, {"do not fail if sampled payloads are copies instead of the original region buffers"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/Logger.h"
#include "Framework/RoutingIndices.h"
#include "DataSampling/DataSamplingHeader.h"
#include "Headers/DataHeader.h"

#include <fairmq/TransportFactory.h>
#include <fairmq/UnmanagedRegion.h>
#include <fairmq/shmem/Message.h>
#include <boost/property_tree/ptree.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

using DataHeader = o2::header::DataHeader;

namespace
{
constexpr uint64_t poisonWord = 0xDEDEDEDEDEDEDEDEull;
constexpr unsigned char poisonByte = 0xDE;
constexpr uint64_t mixer = 0x9E3779B97F4A7C15ull;

uint64_t makeSeed(uint64_t tf, uint64_t part) { return (tf << 16) | part; }

void fillPayload(char* data, size_t size, uint64_t seed)
{
  auto* words = reinterpret_cast<uint64_t*>(data);
  const size_t nWords = size / sizeof(uint64_t);
  words[0] = seed;
  for (size_t i = 1; i < nWords; ++i) {
    words[i] = seed * mixer + i;
  }
}

// returns an empty string if the payload is consistent, otherwise the reason of the failure
std::string checkPayload(const char* data, size_t size, size_t expectedSize, uint32_t subSpec)
{
  if (size != expectedSize) {
    return fmt::format("unexpected size {} (expected {})", size, expectedSize);
  }
  const auto* words = reinterpret_cast<const uint64_t*>(data);
  const size_t nWords = size / sizeof(uint64_t);
  const uint64_t seed = words[0];
  if (seed == poisonWord) {
    return "payload has been poisoned, i.e. released while still in use";
  }
  if ((seed & 0xFFFF) != subSpec) {
    return fmt::format("payload belongs to part {}, but header says subspec {}", seed & 0xFFFF, subSpec);
  }
  for (size_t i = 1; i < nWords; ++i) {
    if (words[i] != seed * mixer + i) {
      return fmt::format("corrupted word {} (value {:#x}{})", i, words[i], words[i] == poisonWord ? ", poisoned" : "");
    }
  }
  return {};
}

/// Free list of buffers of an unmanaged region, filled back by the FairMQ release callback.
struct RegionPool {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<char*> freeBlocks;
  std::atomic<uint64_t> released{0};
  size_t blockSize = 0;
  // declared last, so that it is destroyed first, while the rest of the pool is still alive for its callbacks
  std::unique_ptr<fair::mq::UnmanagedRegion> region;

  char* acquire()
  {
    std::unique_lock lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(20), [this] { return !freeBlocks.empty(); })) {
      return nullptr;
    }
    auto* block = freeBlocks.back();
    freeBlocks.pop_back();
    return block;
  }
};

DataProcessorSpec getProducer(int nTimeframes, int nParts, size_t blockSize, size_t nBlocks)
{
  Outputs outputs;
  for (int p = 0; p < nParts; ++p) {
    outputs.emplace_back(OutputSpec{"TST", "RAWDATA", static_cast<DataHeader::SubSpecificationType>(p)});
  }
  return DataProcessorSpec{
    .name = "stf-like-producer",
    .inputs = {},
    .outputs = outputs,
    .algorithm = adaptStateful([nTimeframes, nParts, blockSize, nBlocks](CallbackService& callbacks) {
      auto pool = std::make_shared<RegionPool>();
      pool->blockSize = blockSize;
      auto tf = std::make_shared<uint64_t>(0);
      // the region has to be destroyed before the transport which created it, like STFSender does
      callbacks.set<CallbackService::Id::Reset>([pool]() {
        LOG(info) << "Destroying the unmanaged region, " << pool->released << " buffers were released";
        pool->region.reset();
      });
      return adaptStateless([pool, tf, nTimeframes, nParts, blockSize, nBlocks](ProcessingContext& ctx) {
        auto* transport = ctx.services().get<FairMQDeviceProxy>().getOutputTransport(RouteIndex{0});
        if (!pool->region) {
          if (transport->GetType() != fair::mq::Transport::SHM) {
            LOG(fatal) << "This test requires the shared memory transport";
          }
          fair::mq::RegionConfig cfg;
          cfg.size = blockSize * nBlocks;
          auto* rawPool = pool.get();
          pool->region = transport->CreateUnmanagedRegion(
            cfg.size,
            [rawPool](const std::vector<fair::mq::RegionBlock>& blocks) {
              // like STFSender: once released, a buffer gets new content. Poisoning it makes any reader which
              // still (wrongly) uses it fail.
              std::lock_guard lock(rawPool->mutex);
              for (const auto& block : blocks) {
                std::memset(block.ptr, poisonByte, rawPool->blockSize);
                rawPool->freeBlocks.push_back(static_cast<char*>(block.ptr));
              }
              rawPool->released += blocks.size();
              rawPool->cv.notify_all();
            },
            cfg);
          auto* base = static_cast<char*>(pool->region->GetData());
          for (size_t b = 0; b < nBlocks; ++b) {
            pool->freeBlocks.push_back(base + b * blockSize);
          }
          LOG(info) << "Created unmanaged region of " << cfg.size << " bytes with " << nBlocks << " buffers";
        }

        if (*tf >= static_cast<uint64_t>(nTimeframes)) {
          LOG(info) << "Produced " << *tf << " timeframes, " << pool->released << " buffers released so far";
          ctx.services().get<ControlService>().endOfStream();
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
          return;
        }
        for (int p = 0; p < nParts; ++p) {
          char* block = pool->acquire();
          if (block == nullptr) {
            LOG(fatal) << "No region buffer was released for 20 seconds, they are leaking";
          }
          fillPayload(block, blockSize, makeSeed(*tf, p));
          auto regionMessage = transport->CreateMessage(pool->region, block, blockSize);
          ctx.outputs().forwardPayload(Output{"TST", "RAWDATA", static_cast<DataHeader::SubSpecificationType>(p)}, *regionMessage);
          // regionMessage is destroyed here, the forwarded shallow copy keeps the buffer alive
        }
        ++(*tf);
      });
    })};
}

DataProcessorSpec getChecker(std::string name, InputSpec input, size_t blockSize, std::chrono::microseconds holdTime, bool sampled, bool expectZeroCopy)
{
  return DataProcessorSpec{
    .name = std::move(name),
    .inputs = {std::move(input)},
    .outputs = {},
    .algorithm = adaptStateful([blockSize, holdTime, sampled, expectZeroCopy](CallbackService& callbacks) {
      struct Stats {
        uint64_t parts = 0;
        uint64_t errors = 0;
        uint64_t inRegion = 0;
        uint64_t notInRegion = 0;
        uint64_t sharedRefs = 0;
      };
      auto stats = std::make_shared<Stats>();
      callbacks.set<CallbackService::Id::EndOfStream>([stats, sampled, expectZeroCopy](EndOfStreamContext&) {
        LOG(info) << "Checked " << stats->parts << " parts, " << stats->errors << " errors, "
                  << stats->inRegion << " in the producer's region, " << stats->notInRegion << " copied, "
                  << stats->sharedRefs << " seen while still shared with another reader";
        if (stats->errors > 0) {
          LOG(fatal) << "Payload verification failed " << stats->errors << " times";
        }
        if (stats->parts == 0) {
          LOG(fatal) << "No data received";
        }
        if (sampled && expectZeroCopy && stats->notInRegion > 0) {
          LOG(fatal) << stats->notInRegion << " sampled payloads were deep copies";
        }
      });
      return adaptStateless([stats, blockSize, holdTime, sampled](ProcessingContext& ctx) {
        for (auto inputIt = ctx.inputs().begin(); inputIt != ctx.inputs().end(); ++inputIt) {
          auto parts = inputIt.parts();
          for (auto partIt = parts.begin(); partIt != parts.end(); ++partIt) {
            const DataRef& ref = *partIt;
            if (ref.header == nullptr) {
              continue;
            }
            const auto* dh = DataRefUtils::getHeader<DataHeader*>(ref);
            if (sampled && DataRefUtils::getHeader<DataSamplingHeader*>(ref) == nullptr) {
              LOG(error) << "Missing DataSamplingHeader";
              stats->errors++;
            }
            const auto size = DataRefUtils::getPayloadSize(ref);
            stats->parts++;

            if (auto* shmMessage = dynamic_cast<fair::mq::shmem::Message*>(parts.getPayloadAtIndices(partIt.indices()))) {
              shmMessage->GetMeta().fManaged ? stats->notInRegion++ : stats->inRegion++;
              if (shmMessage->GetRefCount() > 1) {
                stats->sharedRefs++;
              }
            } else {
              stats->notInRegion++;
            }

            // read twice, holding the message in between, so that the other readers work on it at the same time
            for (int pass = 0; pass < 2; ++pass) {
              if (auto failure = checkPayload(ref.payload, size, blockSize, dh->subSpecification); !failure.empty()) {
                LOG(error) << "Verification of " << dh->dataOrigin.as<std::string>() << "/" << dh->dataDescription.as<std::string>()
                           << "/" << dh->subSpecification << " failed (pass " << pass << "): " << failure;
                stats->errors++;
                break;
              }
              if (pass == 0) {
                std::this_thread::sleep_for(holdTime);
              }
            }
          }
        }
      });
    })};
}
} // namespace

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const int nTimeframes = config.options().get<int>("timeframes");
  const int nParts = config.options().get<int>("parts");
  const size_t blockSize = config.options().get<int>("block-size");
  const size_t nBlocks = config.options().get<int>("blocks");
  const bool expectZeroCopy = !config.options().get<bool>("allow-deep-copy");

  WorkflowSpec specs;
  specs.push_back(getProducer(nTimeframes, nParts, blockSize, nBlocks));
  specs.push_back(getChecker("direct-consumer", InputSpec{"raw", ConcreteDataTypeMatcher{"TST", "RAWDATA"}}, blockSize, std::chrono::microseconds(500), false, expectZeroCopy));
  specs.push_back(getChecker("sampled-consumer-all", InputSpec{"raw", ConcreteDataTypeMatcher{"DS", "sharedAll0"}, Lifetime::Sporadic}, blockSize, std::chrono::milliseconds(2), true, expectZeroCopy));
  specs.push_back(getChecker("sampled-consumer-half", InputSpec{"raw", ConcreteDataTypeMatcher{"DS", "sharedHalf0"}, Lifetime::Sporadic}, blockSize, std::chrono::milliseconds(5), true, expectZeroCopy));

  // two policies on the same data: the Dispatcher forwards the same input message twice
  auto makePolicy = [](const std::string& id, const std::string& fraction) {
    boost::property_tree::ptree condition;
    condition.put("condition", "random");
    condition.put("fraction", fraction);
    condition.put("seed", "1234");
    boost::property_tree::ptree conditions;
    conditions.push_back({"", condition});
    boost::property_tree::ptree policy;
    policy.put("id", id);
    policy.put("active", "true");
    policy.put("query", "raw:TST/RAWDATA");
    policy.add_child("samplingConditions", conditions);
    policy.put("blocking", "true");
    return policy;
  };
  boost::property_tree::ptree policies;
  policies.push_back({"", makePolicy("sharedAll", "1.0")});
  policies.push_back({"", makePolicy("sharedHalf", "0.5")});
  DataSampling::GenerateInfrastructure(specs, policies);

  return specs;
}
