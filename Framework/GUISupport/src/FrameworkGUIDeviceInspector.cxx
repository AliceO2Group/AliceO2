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

#include "FrameworkGUIDeviceInspector.h"
#include "Framework/DataProcessorInfo.h"

#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/ChannelSpec.h"
#include "Framework/Logger.h"
#include "Framework/DeviceController.h"
#include <DebugGUI/icons_font_awesome.h>

#include "DebugGUI/imgui.h"
#include <cinttypes>
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

void deviceInfoTable(char const* label, Metric2DViewIndex const& index, DeviceMetricsInfo const& metrics)
{
  if (index.indexes.empty() == false && ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
    for (size_t i = 0; i < index.indexes.size(); ++i) {
      auto& metric = metrics.metrics[index.indexes[i]];
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
          case VariantType::Int8:
            ImGui::Text("%d (default)", option.defaultValue.get<int8_t>());
            break;
          case VariantType::Int16:
            ImGui::Text("%d (default)", option.defaultValue.get<int16_t>());
            break;
          case VariantType::Int64:
            ImGui::Text("%" PRId64 " (default)", option.defaultValue.get<int64_t>());
            break;
          case VariantType::UInt8:
            ImGui::Text("%d (default)", option.defaultValue.get<uint8_t>());
            break;
          case VariantType::UInt16:
            ImGui::Text("%d (default)", option.defaultValue.get<uint16_t>());
            break;
          case VariantType::UInt32:
            ImGui::Text("%d (default)", option.defaultValue.get<uint32_t>());
            break;
          case VariantType::UInt64:
            ImGui::Text("%" PRIu64 " (default)", option.defaultValue.get<uint64_t>());
            break;
          case VariantType::Float:
            ImGui::Text("%f (default)", option.defaultValue.get<float>());
            break;
          case VariantType::Double:
            ImGui::Text("%f (default)", option.defaultValue.get<double>());
            break;
          case VariantType::Empty:
            ImGui::TextUnformatted(""); // no default value
            break;
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
      if (!service.name.empty()) {
        ImGui::TextUnformatted(service.name.c_str());
      } else {
        ImGui::TextUnformatted("unknown");
      }
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
  ImGui::Text("Device state: %s", info.deviceState.data());
#ifdef DPL_ENABLE_TRACING
  ImGui::Text("Tracy Port: %d", info.tracyPort);
#endif
  ImGui::Text("Rank: %zu/%zu%%%zu/%zu", spec.rank, spec.nSlots, spec.inputTimesliceId, spec.maxInputTimeslices);

  if (ImGui::Button(ICON_FA_BUG "Attach debugger")) {
    std::string pid = std::to_string(info.pid);
    setenv("O2DEBUGGEDPID", pid.c_str(), 1);
#ifdef __APPLE__
    std::string defaultAppleDebugCommand =
      "osascript -e 'tell application \"Terminal\" to activate'"
      " -e 'tell application \"Terminal\" to do script \"lldb -p \" & (system attribute \"O2DEBUGGEDPID\") & \"; exit\"'";
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
    auto defaultAppleProfileCommand = fmt::format(
      "osascript -e 'tell application \"Terminal\"'"
      " -e 'activate'"
      " -e 'do script \"xcrun xctrace record --output dpl-profile-{0}.trace"
      " --time-limit 30s --template Time\\\\ Profiler --attach {0} "
      " && open dpl-profile-{0}.trace && exit\"'"
      " -e 'end tell'",
      pid);

    setenv("O2DPLPROFILE", defaultAppleProfileCommand.c_str(), 0);
#else
    setenv("O2DPLPROFILE", "xterm -hold -e perf record -a -g -p $O2PROFILEDPID > perf-$O2PROFILEDPID.data &", 0);
#endif
    LOG(error) << getenv("O2DPLPROFILE");
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
  if (control.controller) {
    if (ImGui::Button("Offer SHM")) {
      control.controller->write("/shm-offer 1000", strlen("/shm-offer 1000"));
    }

    if (ImGui::Button("Restart")) {
      control.controller->write("/restart", strlen("/restart"));
    }
  }

  deviceInfoTable("Inputs:", info.queriesViewIndex, metrics);
  deviceInfoTable("Outputs:", info.outputsViewIndex, metrics);
  configurationTable(info.currentConfig, info.currentProvenance);
  optionsTable("Workflow Options", metadata.workflowOptions, control);
  servicesTable("Services", spec.services);
  if (ImGui::CollapsingHeader("Command line arguments", ImGuiTreeNodeFlags_DefaultOpen)) {
    static ImGuiTextFilter filter;
    filter.Draw(ICON_FA_SEARCH);
    for (auto& arg : metadata.cmdLineArgs) {
      if (filter.PassFilter(arg.c_str())) {
        ImGui::TextUnformatted(arg.c_str());
      }
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

  bool flagsChanged = false;
  if (ImGui::CollapsingHeader("Event loop tracing", ImGuiTreeNodeFlags_DefaultOpen)) {
    flagsChanged |= ImGui::CheckboxFlags("METRICS_MUST_FLUSH", &control.tracingFlags, DeviceState::LoopReason::METRICS_MUST_FLUSH);
    flagsChanged |= ImGui::CheckboxFlags("SIGNAL_ARRIVED", &control.tracingFlags, DeviceState::LoopReason::SIGNAL_ARRIVED);
    flagsChanged |= ImGui::CheckboxFlags("DATA_SOCKET_POLLED", &control.tracingFlags, DeviceState::LoopReason::DATA_SOCKET_POLLED);
    flagsChanged |= ImGui::CheckboxFlags("DATA_INCOMING", &control.tracingFlags, DeviceState::LoopReason::DATA_INCOMING);
    flagsChanged |= ImGui::CheckboxFlags("DATA_OUTGOING", &control.tracingFlags, DeviceState::LoopReason::DATA_OUTGOING);
    flagsChanged |= ImGui::CheckboxFlags("WS_COMMUNICATION", &control.tracingFlags, DeviceState::LoopReason::WS_COMMUNICATION);
    flagsChanged |= ImGui::CheckboxFlags("TIMER_EXPIRED", &control.tracingFlags, DeviceState::LoopReason::TIMER_EXPIRED);
    flagsChanged |= ImGui::CheckboxFlags("WS_CONNECTED", &control.tracingFlags, DeviceState::LoopReason::WS_CONNECTED);
    flagsChanged |= ImGui::CheckboxFlags("WS_CLOSING", &control.tracingFlags, DeviceState::LoopReason::WS_CLOSING);
    flagsChanged |= ImGui::CheckboxFlags("WS_READING", &control.tracingFlags, DeviceState::LoopReason::WS_READING);
    flagsChanged |= ImGui::CheckboxFlags("WS_WRITING", &control.tracingFlags, DeviceState::LoopReason::WS_WRITING);
    flagsChanged |= ImGui::CheckboxFlags("ASYNC_NOTIFICATION", &control.tracingFlags, DeviceState::LoopReason::ASYNC_NOTIFICATION);
    flagsChanged |= ImGui::CheckboxFlags("OOB_ACTIVITY", &control.tracingFlags, DeviceState::LoopReason::OOB_ACTIVITY);
    flagsChanged |= ImGui::CheckboxFlags("UNKNOWN", &control.tracingFlags, DeviceState::LoopReason::UNKNOWN);
    flagsChanged |= ImGui::CheckboxFlags("FIRST_LOOP", &control.tracingFlags, DeviceState::LoopReason::FIRST_LOOP);
    flagsChanged |= ImGui::CheckboxFlags("NEW_STATE_PENDING", &control.tracingFlags, DeviceState::LoopReason::NEW_STATE_PENDING);
    flagsChanged |= ImGui::CheckboxFlags("PREVIOUSLY_ACTIVE", &control.tracingFlags, DeviceState::LoopReason::PREVIOUSLY_ACTIVE);
    flagsChanged |= ImGui::CheckboxFlags("TRACE_CALLBACKS", &control.tracingFlags, DeviceState::LoopReason::TRACE_CALLBACKS);
    flagsChanged |= ImGui::CheckboxFlags("TRACE_USERCODE", &control.tracingFlags, DeviceState::LoopReason::TRACE_USERCODE);
    if (flagsChanged && control.controller) {
      std::string cmd = fmt::format("/trace {}", control.tracingFlags);
      control.controller->write(cmd.c_str(), cmd.size());
    }
  }
}

} // namespace o2::framework::gui
