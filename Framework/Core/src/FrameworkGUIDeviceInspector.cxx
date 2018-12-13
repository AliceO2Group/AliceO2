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

#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/ChannelSpec.h"

#include "DebugGUI/imgui.h"
#include <csignal>

namespace o2
{
namespace framework
{
namespace gui
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
    }
  }
}

void optionsTable(const DeviceSpec& spec, const DeviceControl& control)
{
  if (spec.options.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader("Options:", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Columns(2);
    auto labels = { "Name", "Value" };
    for (auto& label : labels) {
      ImGui::TextUnformatted(label);
      ImGui::NextColumn();
    }
    for (auto& option : spec.options) {
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

void displayDeviceInspector(DeviceSpec const& spec, DeviceInfo const& info, DeviceMetricsInfo const& metrics, DeviceControl& control)
{
  ImGui::Text("Name: %s", spec.name.c_str());
  ImGui::Text("Pid: %d", info.pid);

  deviceInfoTable(info, metrics);
  optionsTable(spec, control);
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

} // namespace gui
} // namespace framework
} // namespace o2
