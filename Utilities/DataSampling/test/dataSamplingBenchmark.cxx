// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamSpec.h"
#include "DataSampling/DataSampling.h"
#include "Framework/CompletionPolicyHelpers.h"
#include <vector>
#include <filesystem>

using namespace o2::framework;
using namespace o2::utilities;

void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
  policies.push_back(CompletionPolicyHelpers::defineByName("dataSink", CompletionPolicy::CompletionOp::Consume));
}

void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"sampling-fraction", VariantType::Double, 1.0, {"sampling fraction"}});
  workflowOptions.push_back(ConfigParamSpec{"payload-size", VariantType::Int, 10000, {"payload size"}});
  workflowOptions.push_back(ConfigParamSpec{"producers", VariantType::Int, 1, {"number of producers"}});
  workflowOptions.push_back(ConfigParamSpec{"dispatchers", VariantType::Int, 1, {"number of dispatchers"}});
  workflowOptions.push_back(ConfigParamSpec{"usleep", VariantType::Int, 0, {"usleep time of producers"}});
  workflowOptions.push_back(ConfigParamSpec{
    "test-duration", VariantType::Int, 300, {"how long should the test run (in seconds, max. 2147)"}});
  workflowOptions.push_back(
    ConfigParamSpec{"throttling", VariantType::Int, 0, {"stop producing messages if freeram < throttling * 1MB"}});
  workflowOptions.push_back(ConfigParamSpec{
    "fill", VariantType::Bool, false, {"should fill the messages (prevents memory overcommitting)"}});
}

#include <memory>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/functional/hash.hpp>
#include <fairmq/Device.h>
#if defined(__APPLE__)
struct sysinfo {
  unsigned long freeram = -1;
};
void sysinfo(sysinfo* info)
{
  info->freeram = -1;
}
#else
#include <sys/sysinfo.h>
#endif

#include "Headers/DataHeader.h"
#include "Framework/ControlService.h"
#include "DataSampling/DataSampling.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "Framework/RawDeviceService.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;
using namespace o2::utilities;
using SubSpec = o2::header::DataHeader::SubSpecificationType;

namespace bipc = ::boost::interprocess;

// fixme: This is from fairmq/shmem/Common.h (it is not public), try to find a more maintainable solution
inline std::string buildShmIdFromSessionIdAndUserId(const std::string& sessionId)
{
  boost::hash<std::string> stringHash;
  std::string shmId(std::to_string(stringHash(std::string((std::to_string(geteuid()) + sessionId)))));
  shmId.resize(8, '_');
  return shmId;
}

std::function<size_t(void)> createFreeMemoryGetter(InitContext& ictx)
{
  std::string sessionID;
  std::string channelConfig;
  if (fair::mq::Device* device = ictx.services().get<RawDeviceService>().device()) {
    if (auto options = device->GetConfig()) {
      sessionID = options->GetPropertyAsString("session", "");
      channelConfig = options->GetPropertyAsString("channel-config", "");
    }
  }
  if (!sessionID.empty() && sessionID != "default" && channelConfig.find("transport=shmem") != std::string::npos) {
    LOG(info) << "The benchmark is running with shared memory,"
                 " producing messages will be throttled by looking at available memory in the segment: "
              << sessionID;
    std::string segmentID = "fmq_" + buildShmIdFromSessionIdAndUserId(sessionID) + "_main";
    auto segment = std::make_shared<bipc::managed_shared_memory>(bipc::open_only, segmentID.c_str());

    return [segment]() {
      return segment->get_free_memory();
    };
  } else {
#if defined(__APPLE__)
    LOG(warning) << "The benchmark is running without shared memory. "
                    "The throttling mechanism is not supported on MacOS for the ZeroMQ transport, "
                    "the results might be incorrect for larger payload sizes.";
#else
    LOG(info) << "The benchmark is running without shared memory,"
                 " producing messages will be throttled by looking at the global free RAM";
#endif

    auto sysInfo = std::make_shared<struct sysinfo>();

    return [sysInfo]() {
      sysinfo(sysInfo.get());
      return sysInfo->freeram;
    };
  }
}
// clang-format off
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  double samplingFraction = config.options().get<double>("sampling-fraction");
  size_t payloadSize = config.options().get<int>("payload-size");
  size_t producers = config.options().get<int>("producers");
  size_t dispatchers = config.options().get<int>("dispatchers");
  size_t usleepTime = config.options().get<int>("usleep");
  size_t testDuration = config.options().get<int>("test-duration");
  size_t throttlingMB = config.options().get<int>("throttling");
  bool fill = config.options().get<bool>("fill");

  std::string configurationPath = "/tmp/dataSamplingBenchmark-" + std::to_string(samplingFraction) + ".json";
  std::string configuration =
    "{\n"
    "  \"dataSamplingPolicies\": [\n"
    "    {\n"
    "      \"id\": \"benchmark\",\n"
    "      \"active\": \"true\",\n"
    "      \"machines\": [],\n"
    "      \"query\": \"TST:TST/RAWDATA\",\n"
    "      \"samplingConditions\": [\n"
    "        {\n"
    "          \"condition\": \"random\",\n"
    "          \"fraction\": \"" + std::to_string(samplingFraction) + "\",\n"
    "          \"seed\": \"22222\"\n"
    "        }\n"
    "      ],\n"
    "      \"blocking\": \"false\"\n"
    "    }\n"
    "  ]\n"
    "}";

  if (!std::filesystem::exists(configurationPath)) {
    std::ofstream configurationFile(configurationPath);
    configurationFile << configuration;
    configurationFile.close();
  }

  WorkflowSpec specs;

  for (size_t p = 0; p < producers; p++) {
    specs.push_back(DataProcessorSpec{
      "dataProducer" + std::to_string(p),
      Inputs{},
      Outputs{
        OutputSpec{ "TST", "RAWDATA", static_cast<SubSpec>(p) }
      },
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [=](InitContext& ictx) {

          sleep(5); // wait a few seconds before trying to open a shmem segment to make sure it is there

          std::function<size_t(void)> getFreeMemory = createFreeMemoryGetter(ictx);
          const size_t maxFreeMemory = getFreeMemory(); // that may vary on the process sync
          size_t maximumAllowedMessages = (maxFreeMemory - throttlingMB * 1000000) / payloadSize / producers;
          LOG(info) << "First cycle, this producer will send " << maximumAllowedMessages << " messages";

          auto messagesProducedSinceLastCycle = std::make_shared<size_t>(0);
          std::shared_ptr<bool> mightSaturate = nullptr;

          return (AlgorithmSpec::ProcessCallback) [=](ProcessingContext& pctx) mutable {
            usleep(usleepTime);

            // This is a mechanism which protects the benchmark from reaching the maximum of available memory.
            // In that case it is likely that we get killed by oom-killer or trapped inside a deadlock.
            // It works well for shmem tests, but not for zeromq - the latter increases the memory usage
            // muuuch more inertia (slower and harder to control) and we cannot observe it this way.
            // Then, it is recommended to use the '--fill' parameter, which prevents Linux from overcommitting
            // the memory by writing on the produced messages before sending (which unfortunately slows down the
            // message production rate).
            if (*messagesProducedSinceLastCycle < maximumAllowedMessages) {
              auto data = pctx.outputs().make<char>(Output{ "TST", "RAWDATA", static_cast<SubSpec>(p) }, payloadSize);
              *messagesProducedSinceLastCycle += 1;
              if (fill) {
                memset(data.data(), 0x00, payloadSize);
              }
            } else if (mightSaturate == nullptr) {
              // maximumAllowedMessages has been reached, so we check if we should protect the benchmark from asking
              // for too much memory.
              sleep(1);
              // 4 is a magic number - an arbitrary high limit of free memory is 4 times larger than the low limit.
              mightSaturate = std::make_shared<bool>(getFreeMemory() < 4 * throttlingMB * 1000000);
            } else if (*mightSaturate == false) {
              // there is no risk of reaching the maximum available memory,
              // so we allow for a continuous message production.
              *messagesProducedSinceLastCycle = 0;
              maximumAllowedMessages = -1;
              LOG(info) << "The memory usage should not reach the limits, we allow to produce as much messages as possible";
            } else if (size_t freeMemory = getFreeMemory(); freeMemory > 4 * throttlingMB * 1000000) {
              // if we are here, then the maximumAllowedMessages has been reached and we have waited until
              // the memory usage dropped to the safe level again.
              *messagesProducedSinceLastCycle = 0;
              maximumAllowedMessages = (freeMemory - throttlingMB * 1000000) / payloadSize / producers;
              LOG(info) << "New cycle, this producer will send " << maximumAllowedMessages << " messages";
            }
          };
        }
      }
    });
  }

  DataSampling::GenerateInfrastructure(specs, "json:/" + configurationPath, dispatchers);

  DataProcessorSpec podDataSink{
    "dataSink",
    Inputs{{"test-data", {DataSamplingPolicy::createPolicyDataOrigin(), DataSamplingPolicy::createPolicyDataDescription("benchmark", 0)}},
           {"test-timer", "TST", "TIMER", 0, Lifetime::Timer}},
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext & ctx){
        // todo: maybe add a check
        if (ctx.inputs().isValid("test-timer")){
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
        }
      }
    },
    Options{
      { "period-test-timer", VariantType::Int, static_cast<int>(testDuration * 1000000), { "timer period" }}
    }
  };

  specs.push_back(podDataSink);
  return specs;
}
// clang-format on
