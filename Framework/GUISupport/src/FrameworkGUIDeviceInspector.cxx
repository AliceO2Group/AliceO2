// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FrameworkGUIDeviceInspector.h"
#include "Framework/DataProcessorInfo.h"

#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/ChannelSpec.h"
#include "Framework/Logger.h"

#include "DebugGUI/imgui.h"
#include <csignal>
#include <cstdlib>
#include <iostream>

namespace o2::framework::gui
{

struct ChannelsTableHelper {
  template <typename C>
  static void channelsTable(const char* title, const C& channels)
  {
    ImGui::TextUnformatted(title);
    ImGui::Columns(2);
    ImGui::TextUnformatted("Name");
    ImGui::NextColumn();
    ImGui::TextUnformatted("Port");
    ImGui::NextColumn();
    for (auto channel : channels) {
      ImGui::TextUnformatted(channel.name.c_str());
      ImGui::NextColumn();
      ImGui::Text("%d", channel.port);
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
};

void deviceInfoTable(DeviceInfo const& info, DeviceMetricsInfo const& metrics)
{
  if (info.queriesViewIndex.indexes.empty() == false && ImGui::CollapsingHeader("Inputs:", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (size_t i = 0; i < info.queriesViewIndex.indexes.size(); ++i) {
      auto& metric = metrics.metrics[info.queriesViewIndex.indexes[i]];
      ImGui::Text("%zu: %s", i, metrics.stringMetrics[metric.storeIdx][0].data);
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("%zu: %s", i, metrics.stringMetrics[metric.storeIdx][0].data);
        ImGui::EndTooltip();
      }
    }
  }
}

void configurationTable(boost::property_tree::ptree const& currentConfig,
                        boost::property_tree::ptree const& currentProvenance)
{
  if (currentConfig.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader("Current Config", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Columns(2);
    auto labels = {"Name", "Value"};
    for (auto& label : labels) {
      ImGui::TextUnformatted(label);
      ImGui::NextColumn();
    }
    for (auto& option : currentConfig) {
      ImGui::TextUnformatted(option.first.c_str());
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(option.first.c_str());
        ImGui::EndTooltip();
      }
      ImGui::NextColumn();
      ImGui::Text("%s (%s)", option.second.data().c_str(), currentProvenance.get<std::string>(option.first).c_str());
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("%s (%s)", option.second.data().c_str(), currentProvenance.get<std::string>(option.first).c_str());
        ImGui::EndTooltip();
      }
      ImGui::NextColumn();
    }
    ImGui::Columns(1);
  }
}

void optionsTable(const char* label, std::vector<ConfigParamSpec> const& options, const DeviceControl& control)
{
  if (options.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Columns(2);
    auto labels = {"Name", "Value"};
    for (auto& label : labels) {
      ImGui::TextUnformatted(label);
      ImGui::NextColumn();
    }
    for (auto& option : options) {
      ImGui::TextUnformatted(option.name.c_str());
      ImGui::NextColumn();
      auto currentValueIt = control.options.find(option.name);

      // Did not find the option
      if (currentValueIt == control.options.end()) {
        switch (option.type) {
          case VariantType::String:
            ImGui::Text(R"("%s" (default))", option.defaultValue.get<const char*>());
            break;
          case VariantType::Int:
            ImGui::Text("%d (default)", option.defaultValue.get<int>());
            break;
          case VariantType::Float:
            ImGui::Text("%f (default)", option.defaultValue.get<float>());
            break;
          case VariantType::Double:
            ImGui::Text("%f (default)", option.defaultValue.get<double>());
            break;
          case VariantType::Empty:
            ImGui::TextUnformatted(""); // no default value
          default:
            ImGui::TextUnformatted("unknown");
        }
      } else {
        ImGui::TextUnformatted(currentValueIt->second.c_str());
      }
      ImGui::NextColumn();
    }
  }

  ImGui::Columns(1);
}

void servicesTable(const char* label, std::vector<ServiceSpec> const& services)
{
  if (services.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Columns(2);
    auto labels = {"Service", "Kind"};
    for (auto& label : labels) {
      ImGui::TextUnformatted(label);
      ImGui::NextColumn();
    }
    for (auto& service : services) {
      ImGui::TextUnformatted(service.name.c_str());
      ImGui::NextColumn();
      switch (service.kind) {
        case ServiceKind::Serial:
          ImGui::TextUnformatted("Serial");
          break;
        case ServiceKind::Global:
          ImGui::TextUnformatted("Global");
          break;
        case ServiceKind::Stream:
          ImGui::TextUnformatted("Stream");
          break;
        default:
          ImGui::TextUnformatted("unknown");
      }
      ImGui::NextColumn();
    }
  }

  ImGui::Columns(1);
}

void displayDeviceInspector(DeviceSpec const& spec,
                            DeviceInfo const& info,
                            DeviceMetricsInfo const& metrics,
                            DataProcessorInfo const& metadata,
                            DeviceControl& control)
{
  ImGui::Text("Name: %s", spec.name.c_str());
  ImGui::Text("Executable: %s", metadata.executable.c_str());
  if (info.active) {
    ImGui::Text("Pid: %d", info.pid);
  } else {
    ImGui::Text("Pid: %d (exit status: %d)", info.pid, info.exitStatus);
  }
#ifdef DPL_ENABLE_TRACING
  ImGui::Text("Tracy Port: %d", info.tracyPort);
#endif
  ImGui::Text("Rank: %zu/%zu%%%zu/%zu", spec.rank, spec.nSlots, spec.inputTimesliceId, spec.maxInputTimeslices);

  if (ImGui::Button("Attach debugger")) {
    std::string pid = std::to_string(info.pid);
    setenv("O2DEBUGGEDPID", pid.c_str(), 1);
#ifdef __APPLE__
    std::string defaultAppleDebugCommand =
      "osascript -e 'tell application \"Terminal\" to activate'"
      " -e 'tell application \"Terminal\" to do script \"lldb -p \" & (system attribute \"O2DEBUGGEDPID\")'";
    setenv("O2DPLDEBUG", defaultAppleDebugCommand.c_str(), 0);
#else
    setenv("O2DPLDEBUG", "xterm -hold -e gdb attach $O2DEBUGGEDPID &", 0);
#endif
    int retVal = system(getenv("O2DPLDEBUG"));
    (void)retVal;
  }

  ImGui::SameLine();
  if (ImGui::Button("Profile 30s")) {
    std::string pid = std::to_string(info.pid);
    setenv("O2PROFILEDPID", pid.c_str(), 1);
#ifdef __APPLE__
    std::string defaultAppleProfileCommand =
      "osascript -e 'tell application \"Terminal\" to activate'"
      " -e 'tell application \"Terminal\" to do script \"instruments -D dpl-profile-" +
      pid +
      ".trace -l 30000 -t Time\\\\ Profiler -p " +
      pid + " && open dpl-profile-" + pid + ".trace && exit\"'";
    setenv("O2DPLPROFILE", defaultAppleProfileCommand.c_str(), 0);
#else
    setenv("O2DPLPROFILE", "xterm -hold -e perf record -a -g -p $O2PROFILEDPID > perf-$O2PROFILEDPID.data &", 0);
#endif
    LOG(ERROR) << getenv("O2DPLPROFILE");
    int retVal = system(getenv("O2DPLPROFILE"));
    (void)retVal;
  }

#if DPL_ENABLE_TRACING
  ImGui::SameLine();
  if (ImGui::Button("Tracy")) {
    std::string tracyPort = std::to_string(info.tracyPort);
    auto cmd = fmt::format("tracy-profiler -p {} -a 127.0.0.1 &", info.tracyPort);
    LOG(debug) << cmd;
    int retVal = system(cmd.c_str());
    (void)retVal;
  }
#endif

  deviceInfoTable(info, metrics);
  for (auto& option : info.currentConfig) {
    ImGui::Text("%s: %s", option.first.c_str(), option.second.data().c_str());
  }
  configurationTable(info.currentConfig, info.currentProvenance);
  optionsTable("Workflow Options", metadata.workflowOptions, control);
  servicesTable("Services", spec.services);
  if (ImGui::CollapsingHeader("Command line arguments", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (auto& arg : metadata.cmdLineArgs) {
      ImGui::Text("%s", arg.c_str());
    }
  }

  if (ImGui::CollapsingHeader("Channels", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("# channels: %lu", spec.inputChannels.size() + spec.outputChannels.size());
    ChannelsTableHelper::channelsTable("Inputs:", spec.inputChannels);
    ChannelsTableHelper::channelsTable("Outputs:", spec.outputChannels);
  }
  if (ImGui::CollapsingHeader("Data relayer")) {
    ImGui::Text("Completion policy: %s", spec.completionPolicy.name.c_str());
  }
  if (ImGui::CollapsingHeader("Signals", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Button("SIGSTOP")) {
      kill(info.pid, SIGSTOP);
    }
    ImGui::SameLine();
    if (ImGui::Button("SIGTERM")) {
      kill(info.pid, SIGTERM);
    }
    ImGui::SameLine();
    if (ImGui::Button("SIGKILL")) {
      kill(info.pid, SIGKILL);
    }
    if (ImGui::Button("SIGCONT")) {
      kill(info.pid, SIGCONT);
    }
    ImGui::SameLine();
    if (ImGui::Button("SIGUSR1")) {
      kill(info.pid, SIGUSR1);
    }
    ImGui::SameLine();
    if (ImGui::Button("SIGUSR2")) {
      kill(info.pid, SIGUSR2);
    }
  }
}

} // namespace o2::framework::gui
