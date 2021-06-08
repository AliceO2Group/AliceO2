// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "O2ControlHelpers.h"
#include "Framework/O2ControlLabels.h"
#include "ChannelSpecHelpers.h"
#include "Framework/Logger.h"

#include <iostream>
#include <cstring>
#include <string>
#include <filesystem>

namespace bfs = std::filesystem;

namespace o2::framework
{

const char* indScheme = "  ";

namespace implementation
{

std::string taskName(const std::string& workflowName, const std::string& deviceName)
{
  return workflowName + "-" + deviceName;
}

template <typename T>
void dumpChannelBind(std::ostream& dumpOut, const T& channel, std::string indLevel)
{
  dumpOut << indLevel << "- name: " << channel.name << "\n";
  dumpOut << indLevel << indScheme << "type: " << ChannelSpecHelpers::typeAsString(channel.type) << "\n";
  // todo: i shouldn't guess here
  dumpOut << indLevel << indScheme << "transport: " << (channel.protocol == ChannelProtocol::IPC ? "shmem" : "zeromq") << "\n";
  dumpOut << indLevel << indScheme << "addressing: " << (channel.protocol == ChannelProtocol::IPC ? "ipc" : "tcp") << "\n";
  dumpOut << indLevel << indScheme << "rateLogging: \"{{ fmq_rate_logging }}\"\n";
}

template <typename T>
void dumpChannelConnect(std::ostream& dumpOut, const T& channel, const std::string& binderName, std::string indLevel)
{
  dumpOut << indLevel << "- name: " << channel.name << "\n";
  dumpOut << indLevel << indScheme << "type: " << ChannelSpecHelpers::typeAsString(channel.type) << "\n";
  // todo: i shouldn't guess here
  dumpOut << indLevel << indScheme << "transport: " << (channel.protocol == ChannelProtocol::IPC ? "shmem" : "zeromq") << "\n";
  dumpOut << indLevel << indScheme << "target: \"{{ Parent().Path }}." << binderName << ":" << channel.name << "\"\n";
  dumpOut << indLevel << indScheme << "rateLogging: \"{{ fmq_rate_logging }}\"\n";
}

struct RawChannel {
  std::string_view name;
  std::string_view type;
  std::string_view method;
  std::string_view address;
  std::string_view rateLogging;
  std::string_view transport;
};

std::string rawChannelReference(std::string_view channelName, bool isUniqueChannel)
{
  if (isUniqueChannel) {
    return std::string(channelName);
  } else {
    return std::string(channelName) + "-{{ it }}";
  }
}

void dumpRawChannelConnect(std::ostream& dumpOut, const RawChannel& channel, bool isUniqueChannel, std::string indLevel)
{
  auto channelRef = rawChannelReference(channel.name, isUniqueChannel);
  LOG(INFO) << "This topology will connect to the channel '" << channel.name << "', which is most likely bound outside."
            << " Please make sure it is declared in the global channel space under the name '" << channelRef
            << "' in the mother workflow or another subworkflow.";

  dumpOut << indLevel << "- name: " << channel.name << "\n";
  dumpOut << indLevel << indScheme << "type: " << channel.type << "\n";
  dumpOut << indLevel << indScheme << "transport: " << channel.transport << "\n";
  dumpOut << indLevel << indScheme << "target: \"::" << channelRef << "\"\n";
  dumpOut << indLevel << indScheme << "rateLogging: \"{{ fmq_rate_logging }}\"\n";
}

void dumpRawChannelBind(std::ostream& dumpOut, const RawChannel& channel, bool isUniqueChannel, std::string indLevel)
{
  auto channelRef = rawChannelReference(channel.name, isUniqueChannel);
  LOG(INFO) << "This topology will bind a dangling channel '" << channel.name << "'"
            << " and declare it in the global channel space under the name '" << channelRef << "'."
            << " Please make sure that another device connects to this channel elsewhere."
            << " Also, don't mind seeing the message twice, it will be addressed in future releases.";

  dumpOut << indLevel << "- name: " << channel.name << "\n";
  dumpOut << indLevel << indScheme << "type: " << channel.type << "\n";
  dumpOut << indLevel << indScheme << "transport: " << channel.transport << "\n";
  dumpOut << indLevel << indScheme << "addressing: " << (channel.address.find("ipc") != std::string_view::npos ? "ipc" : "tcp") << "\n";
  dumpOut << indLevel << indScheme << "rateLogging: \"{{ fmq_rate_logging }}\"\n";
  dumpOut << indLevel << indScheme << "global: \"" << channelRef << "\"\n";
}

std::string_view extractValueFromChannelConfig(std::string_view string, std::string_view token)
{
  size_t tokenStart = string.find(token);
  if (tokenStart == std::string_view::npos) {
    return {};
  }
  size_t valueStart = tokenStart + token.size();
  if (valueStart >= string.size()) {
    return {};
  }
  size_t valueEnd = string.find(',', valueStart);
  return valueEnd == std::string_view::npos
           ? string.substr(valueStart, string.size() - valueStart)
           : string.substr(valueStart, valueEnd - valueStart);
}

// fixme: For now we extract information about raw FairMQ channels from execution.
//  However, we risk that it break if a channel configuration method changes,
//  thus this information should be provided in DeviceSpec. Find a way to do that.
std::vector<RawChannel> extractRawChannels(const DeviceSpec& spec, const DeviceExecution& execution)
{
  std::vector<std::string> dplChannels;
  for (const auto& channel : spec.inputChannels) {
    dplChannels.emplace_back(channel.name);
  }
  for (const auto& channel : spec.outputChannels) {
    dplChannels.emplace_back(channel.name);
  }

  std::vector<RawChannel> rawChannels;
  for (size_t i = 0; i < execution.args.size(); i++) {
    if (execution.args[i] != nullptr && strcmp(execution.args[i], "--channel-config") == 0 && i + 1 < execution.args.size()) {
      auto channelConfig = execution.args[i + 1];
      auto channelName = extractValueFromChannelConfig(channelConfig, "name=");
      if (std::find(dplChannels.begin(), dplChannels.end(), channelName) == dplChannels.end()) {
        // "name=readout-proxy,type=pair,method=connect,address=ipc:///tmp/readout-pipe-0,rateLogging=1,transport=shmem"
        rawChannels.push_back({channelName,
                               extractValueFromChannelConfig(channelConfig, "type="),
                               extractValueFromChannelConfig(channelConfig, "method="),
                               extractValueFromChannelConfig(channelConfig, "address="),
                               extractValueFromChannelConfig(channelConfig, "rateLogging="),
                               extractValueFromChannelConfig(channelConfig, "transport=")});
      }
    }
  }
  return rawChannels;
}

void dumpCommand(std::ostream& dumpOut, const DeviceExecution& execution, std::string indLevel)
{
  dumpOut << indLevel << "shell: true\n";
  dumpOut << indLevel << "log: \"{{ log_task_output }}\"\n";
  dumpOut << indLevel << "user: \"{{ user }}\"\n";
  dumpOut << indLevel << "value: \"{{ len(modulepath)>0 ? _module_cmdline : _plain_cmdline }}\"\n";

  dumpOut << indLevel << "arguments:\n";
  dumpOut << indLevel << indScheme << "- \"-b\"\n";
  dumpOut << indLevel << indScheme << "- \"--monitoring-backend\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ monitoring_dpl_url }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--session\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ session_id }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--infologger-severity\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ infologger_severity }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--infologger-mode\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ infologger_mode }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--driver-client-backend\"\n";
  dumpOut << indLevel << indScheme << "- \"'stdout://'\"\n";
  dumpOut << indLevel << indScheme << "- \"--shm-segment-size\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ shm_segment_size }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--shm-throw-bad-alloc\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ shm_throw_bad_alloc }}'\"\n";
  dumpOut << indLevel << indScheme << "- \"--resources-monitoring\"\n";
  dumpOut << indLevel << indScheme << "- \"'{{ resources_monitoring }}'\"\n";

  for (size_t ai = 1; ai < execution.args.size(); ++ai) {
    const char* option = execution.args[ai];
    const char* value = nullptr; // no value by default (i.e. a boolean)
    // If the subsequent option exists and does not start with -, we assume
    // it is an argument to the previous one.
    // ...that is unless it is a "-1" for example.
    if (ai + 1 < execution.args.size() && execution.args[ai + 1][0] != '-') {
      value = execution.args[ai + 1];
      ai++;
    }
    if (!option) {
      break;
    }

    static const std::set<std::string> omitOptions = {
      "--channel-config", "--o2-control", "--control", "--session", "--monitoring-backend",
      "-b", "--color", "--infologger-severity", "--infologger-mode", "--driver-client-backend",
      "--shm-segment-size", "--shm-throw-bad-alloc", "--resources-monitoring"};
    if (omitOptions.find(option) != omitOptions.end()) {
      continue;
    }
    // todo: possible improvement - do not print if default values are used
    // todo: check if '' are there already.
    dumpOut << indLevel << indScheme << R"(- ")" << option << "\"\n";
    if (value) {
      dumpOut << indLevel << indScheme << R"(- "')" << value << "'\"\n";
    }
  }
}

std::string findBinder(const std::vector<DeviceSpec>& specs, const std::string& channel)
{
  // fixme: it is not crucial to be fast here, but ideally we should check only input OR output channels.
  for (const auto& spec : specs) {
    for (const auto& inputChannel : spec.inputChannels) {
      if (inputChannel.method == ChannelMethod::Bind && inputChannel.name == channel) {
        return spec.id;
      }
    }
    for (const auto& outputChannel : spec.outputChannels) {
      if (outputChannel.method == ChannelMethod::Bind && outputChannel.name == channel) {
        return spec.id;
      }
    }
  }
  throw std::runtime_error("Could not find a device which binds the '" + channel + "' channel.");
}

bool isUniqueProxy(const DeviceSpec& spec)
{
  return std::find(spec.labels.begin(), spec.labels.end(), ecs::uniqueProxyLabel) != spec.labels.end();
}

void dumpRole(std::ostream& dumpOut, const std::string& taskName, const DeviceSpec& spec, const std::vector<DeviceSpec>& allSpecs, const DeviceExecution& execution, const std::string indLevel)
{
  dumpOut << indLevel << "- name: \"" << spec.id << "\"\n";

  dumpOut << indLevel << indScheme << "connect:\n";

  for (const auto& outputChannel : spec.outputChannels) {
    if (outputChannel.method == ChannelMethod::Connect) {
      dumpChannelConnect(dumpOut, outputChannel, findBinder(allSpecs, outputChannel.name), indLevel + indScheme);
    }
  }
  for (const auto& inputChannel : spec.inputChannels) {
    if (inputChannel.method == ChannelMethod::Connect) {
      dumpChannelConnect(dumpOut, inputChannel, findBinder(allSpecs, inputChannel.name), indLevel + indScheme);
    }
  }
  bool uniqueProxy = isUniqueProxy(spec);
  bool bindsRawChannels = false;
  auto rawChannels = extractRawChannels(spec, execution);
  for (const auto& rawChannel : rawChannels) {
    if (rawChannel.method == "connect") {
      dumpRawChannelConnect(dumpOut, rawChannel, uniqueProxy, indLevel + indScheme);
    } else if (rawChannel.method == "bind") {
      bindsRawChannels = true;
    }
  }

  // for the time being we have to publish global bound channels also in WFT
  if (bindsRawChannels) {
    dumpOut << indLevel << indScheme << "bind:\n";
    for (const auto& rawChannel : rawChannels) {
      if (rawChannel.method == "bind") {
        dumpRawChannelBind(dumpOut, rawChannel, uniqueProxy, indLevel + indScheme);
      }
    }
  }

  dumpOut << indLevel << indScheme << "task:\n";
  dumpOut << indLevel << indScheme << indScheme << "load: " << taskName << "\n";
}

std::string removeO2ControlArg(std::string_view command)
{
  const char* o2ControlArg = " --o2-control ";
  size_t o2ControlArgStart = command.find(o2ControlArg);
  if (o2ControlArgStart == std::string_view::npos) {
    return std::string(command);
  }
  size_t o2ControlArgEnd = command.find(" ", o2ControlArgStart + std::strlen(o2ControlArg));
  auto result = std::string(command.substr(0, o2ControlArgStart));
  if (o2ControlArgEnd != std::string_view::npos) {
    result += command.substr(o2ControlArgEnd);
  }
  return result;
}

} // namespace implementation

void dumpTask(std::ostream& dumpOut, const DeviceSpec& spec, const DeviceExecution& execution, std::string taskName, std::string indLevel)
{
  dumpOut << indLevel << "name: " << taskName << "\n";
  dumpOut << indLevel << "defaults:\n";
  dumpOut << indLevel << indScheme << "log_task_output: none\n";

  if (bfs::path(execution.args[0]).filename().string() != execution.args[0]) {
    LOG(WARNING) << "The workflow template generation was started with absolute or relative executables paths."
                    " Please use the symlinks exported by the build infrastructure or remove the paths manually in the generated templates,"
                    " unless you really need executables within concrete directories";
  }
  dumpOut << indLevel << indScheme << "_module_cmdline: >-\n";
  dumpOut << indLevel << indScheme << indScheme << "source /etc/profile.d/modules.sh && MODULEPATH={{ modulepath }} module load O2 QualityControl Control-OCCPlugin &&\n";
  dumpOut << indLevel << indScheme << indScheme << "{{ dpl_command }} | " << execution.args[0] << "\n";
  dumpOut << indLevel << indScheme << "_plain_cmdline: \"source /etc/profile.d/o2.sh && {{ dpl_command }} | " << execution.args[0] << "\"\n";

  dumpOut << indLevel << "control:\n";
  dumpOut << indLevel << indScheme << "mode: \"fairmq\"\n";

  // todo: find out proper values perhaps...
  dumpOut << indLevel << "wants:\n";
  dumpOut << indLevel << indScheme << "cpu: 0.01\n";
  dumpOut << indLevel << indScheme << "memory: 1\n";

  dumpOut << indLevel << "bind:\n";
  for (const auto& outputChannel : spec.outputChannels) {
    if (outputChannel.method == ChannelMethod::Bind) {
      implementation::dumpChannelBind(dumpOut, outputChannel, indLevel + indScheme);
    }
  }
  for (const auto& inputChannel : spec.inputChannels) {
    if (inputChannel.method == ChannelMethod::Bind) {
      implementation::dumpChannelBind(dumpOut, inputChannel, indLevel + indScheme);
    }
  }
  bool uniqueProxy = implementation::isUniqueProxy(spec);
  auto rawChannels = implementation::extractRawChannels(spec, execution);
  for (const auto& rawChannel : rawChannels) {
    if (rawChannel.method == "bind") {
      dumpRawChannelBind(dumpOut, rawChannel, uniqueProxy, indLevel + indScheme);
    }
  }

  dumpOut << indLevel << "command:\n";
  implementation::dumpCommand(dumpOut, execution, indLevel + indScheme);
}

void dumpWorkflow(std::ostream& dumpOut, const std::vector<DeviceSpec>& specs, const std::vector<DeviceExecution>& executions, const CommandInfo& commandInfo, std::string workflowName, std::string indLevel)
{
  dumpOut << indLevel << "name: " << workflowName << "\n";

  dumpOut << indLevel << "vars:\n";
  dumpOut << indLevel << indScheme << "dpl_command: >-\n";
  dumpOut << indLevel << indScheme << indScheme << implementation::removeO2ControlArg(commandInfo.command) << "\n";

  dumpOut << indLevel << "defaults:\n";
  dumpOut << indLevel << indScheme << "monitoring_dpl_url: \"no-op://\"\n";
  dumpOut << indLevel << indScheme << "user: \"flp\"\n";
  dumpOut << indLevel << indScheme << "fmq_rate_logging: 0\n";
  dumpOut << indLevel << indScheme << "shm_segment_size: 10000000000\n";
  dumpOut << indLevel << indScheme << "shm_throw_bad_alloc: false\n";
  dumpOut << indLevel << indScheme << "session_id: default\n";
  dumpOut << indLevel << indScheme << "resources_monitoring: 15\n";

  dumpOut << indLevel << "roles:\n";
  for (size_t di = 0; di < specs.size(); di++) {
    auto& spec = specs[di];
    auto& execution = executions[di];
    implementation::dumpRole(dumpOut, implementation::taskName(workflowName, spec.id), spec, specs, execution, indLevel + indScheme);
  }
}

void dumpDeviceSpec2O2Control(std::string workflowName,
                              const std::vector<DeviceSpec>& specs,
                              const std::vector<DeviceExecution>& executions,
                              const CommandInfo& commandInfo)
{
  const char* tasksDirectory = "tasks";
  const char* workflowsDirectory = "workflows";

  LOG(INFO) << "Dumping the workflow configuration for AliECS.";

  LOG(INFO) << "Creating directories '" << workflowsDirectory << "' and '" << tasksDirectory << "'.";
  std::filesystem::create_directory(workflowsDirectory);
  std::filesystem::create_directory(tasksDirectory);
  LOG(INFO) << "... created.";

  assert(specs.size() == executions.size());

  LOG(INFO) << "Creating a workflow dump '" + workflowName + "'.";
  std::string wfDumpPath = std::string(workflowsDirectory) + bfs::path::preferred_separator + workflowName + ".yaml";
  std::ofstream wfDump(wfDumpPath);
  dumpWorkflow(wfDump, specs, executions, commandInfo, workflowName, "");
  wfDump.close();

  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    auto& execution = executions[di];

    LOG(INFO) << "Creating a task dump for '" + spec.id + "'.";
    std::string taskName = implementation::taskName(workflowName, spec.id);
    std::string taskDumpPath = std::string(tasksDirectory) + bfs::path::preferred_separator + taskName + ".yaml";
    std::ofstream taskDump(taskDumpPath);
    dumpTask(taskDump, spec, execution, taskName, "");
    taskDump.close();
    LOG(INFO) << "...created.";
  }
}

} // namespace o2::framework
