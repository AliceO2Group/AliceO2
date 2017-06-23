// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FrameworkGUIDebugger.h"
#include "Framework/FrameworkGUIDevicesGraph.h"
#include "DebugGUI/imgui.h"
#include <set>
#include <algorithm>

namespace o2 {
namespace framework {

struct DeviceGUIState {
  std::string label;
};

struct WorkspaceGUIState {
  int selectedMetric;
  std::vector<std::string> availableMetrics;
  std::vector<DeviceGUIState> devices;
};

static WorkspaceGUIState gState;

void
optionsTable(const DeviceSpec &spec, const DeviceControl &control) {
  if (spec.options.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader("Options:")) {
    ImGui::Columns(2);
    auto labels = {"Name", "Value"};
    for (auto &label : labels) {
      ImGui::TextUnformatted(label);
      ImGui::NextColumn();
    }
    for (auto &option : spec.options) {
      ImGui::TextUnformatted(option.name.c_str());
      ImGui::NextColumn();
      auto currentValueIt = control.options.find(option.name);

      // Did not find the option
      if (currentValueIt == control.options.end()) {
        switch(option.type) {
          case VariantType::String:
            ImGui::Text("\"%s\" (default)", option.defaultValue.get<const char *>());
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

void displayHistory(const DeviceInfo &info, DeviceControl &control) {
  if (info.history.empty()) {
    return;
  }
  int startPos = info.historyPos;
  const int historySize = info.history.size();

  int triggerStartPos = startPos + 1 % historySize;
  int triggerStopPos = startPos % historySize;
  bool triggerStart = false;
  bool triggerStop = false;

  int j = startPos;
  // We look for a stop trigger, so that we know where to stop the search for
  // out start search. If no stop trigger is found, we search until the end
  if (control.logStopTrigger[0]) {
    while((j % historySize) != ((startPos + 1) % historySize)) {
      assert(j >= 0);
      assert(j < historySize);
      auto &line = info.history[j];
      if (strstr(line.c_str(), control.logStopTrigger)) {
        triggerStopPos = (j + 1) % historySize;
        break;
      }
      // Wrap in case we end up below 0
      j = (j == 0) ? historySize - 1 : j - 1;
    }
  }

  // Look for the last instance of the start trigger before the
  // last stop trigger.
  j = startPos + 1;
  if (control.logStartTrigger[0]) {
    while((j % historySize) != triggerStopPos) {
      assert(historySize > j);
      assert(historySize == 1000);
      auto &line = info.history[j];
      if (strstr(line.c_str(), control.logStartTrigger)) {
        triggerStartPos = j;
      }
      j = (j + 1) % historySize;
    }
  }

  // We start from the last trigger found. Eventually this is the first
  // line in the ring buffer, if no trigger is specified.
  size_t ji = triggerStartPos % historySize;
  size_t je = triggerStopPos % historySize;
  size_t iterations = 0;
  while (historySize
         && ((ji % historySize) != je)) {
    assert(iterations < historySize);
    iterations++;
    assert(historySize == 1000);
    assert(ji < historySize);
    auto &line = info.history[ji];
    // Skip empty lines
    if (line.empty()) {
      ji = (ji + 1) % historySize;
      continue;
    }
    // Print matching lines
    if (strstr(line.c_str(), control.logFilter) != 0) {
      ImGui::TextUnformatted(line.c_str(), line.c_str() + line.size());
    }
    ji = (ji + 1) % historySize;
  }
}

template <typename T>
struct HistoData {
  int mod;
  size_t first;
  size_t size;
  const T *points;
};


void
historyBar(DeviceGUIState &state,
           const DeviceSpec &spec,
           const DeviceMetricsInfo &metricsInfo) {
  bool open = ImGui::TreeNode(state.label.c_str());
  if (open) {
    ImGui::Text("# channels: %lu", spec.channels.size());
    ImGui::TreePop();
  }
  ImGui::NextColumn();

  if (gState.selectedMetric == -1) {
    ImGui::NextColumn();
    return;
  }

  auto currentMetricName = gState.availableMetrics[gState.selectedMetric];

  size_t i = metricIdxByName(currentMetricName, metricsInfo);
  // We did not find any plot, skipping this.
  if (i == metricsInfo.metricLabelsIdx.size()) {
    ImGui::NextColumn();
    return;
  }
  auto &metric = metricsInfo.metrics[i];

  switch(metric.type) {
    case MetricType::Int:
      {
        HistoData<int> data;
        data.mod = metricsInfo.timestamps[i].size();
        data.first = metric.pos - data.mod;
        data.size = metricsInfo.intMetrics[metric.storeIdx].size();
        data.points = metricsInfo.intMetrics[metric.storeIdx].data();

        auto getter = [](void *hData, int idx) -> float {
          auto histoData = reinterpret_cast<HistoData<int> *>(hData);
          size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
          assert(pos >= 0 && pos < 1024);
          return histoData->points[pos];
        };
        ImGui::PlotLines(currentMetricName.c_str(),
                         getter,
                         &data,
                         data.size);
        ImGui::NextColumn();
      }
    break;
    case MetricType::Float:
      {
        HistoData<float> data;
        data.mod = metricsInfo.timestamps[i].size();
        data.first = metric.pos - data.mod;
        data.size = metricsInfo.floatMetrics[metric.storeIdx].size();
        data.points = metricsInfo.floatMetrics[metric.storeIdx].data();

        auto getter = [](void *hData, int idx) -> float {
          auto histoData = reinterpret_cast<HistoData<float> *>(hData);
          size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
          assert(pos >= 0 && pos < 1024);
          return histoData->points[pos];
        };
        ImGui::PlotLines(currentMetricName.c_str(),
                         getter,
                         &data,
                         data.size);
        ImGui::NextColumn();
      }
    break;
    default:
      ImGui::NextColumn();
      return;
    break;
  }
}

void
displayDeviceHistograms(const std::vector<DeviceInfo> &infos,
                        const std::vector<DeviceSpec> &devices,
                        std::vector<DeviceControl> &controls,
                        const std::vector<DeviceMetricsInfo> &metricsInfos) {
  bool graphNodes = true;
  showTopologyNodeGraph(&graphNodes, infos, devices);
  ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - 300), 0);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 300), 0);

  // Calculate the unique set of metrics, as available in the metrics
  // service
  std::set<std::string> allMetricsNames;
  for (const auto &metricsInfo : metricsInfos) {
    for (const auto &labelsPairs : metricsInfo.metricLabelsIdx) {
      allMetricsNames.insert(labelsPairs.first);
    }
  }
  using NamesIndex = std::vector<std::string>;
  gState.availableMetrics.clear();
  std::copy(allMetricsNames.begin(),
            allMetricsNames.end(),
            std::back_inserter(gState.availableMetrics));

  ImGui::Begin("Devices");
  ImGui::Combo("Select metric",
               &gState.selectedMetric,
               [](void *data, int idx, const char **outText) -> bool {
                 NamesIndex *v = reinterpret_cast<NamesIndex*>(data);
                 if (idx >= v->size()) {
                   return false;
                 }
                 *outText = v->at(idx).c_str();
                 return true;
               },
               &gState.availableMetrics,
               gState.availableMetrics.size()
  );

  // Calculate the full timestamp range for the selected metric
  if (gState.selectedMetric >= 0) {
    auto currentMetricName = gState.availableMetrics[gState.selectedMetric];
    size_t minTime = -1;
    size_t maxTime = 0;
    for (auto &metricInfo : metricsInfos) {
      size_t mi = metricIdxByName(currentMetricName, metricInfo);
      if (mi == metricInfo.metricLabelsIdx.size()) {
        continue;
      }
      auto &metric = metricInfo.metrics[mi];
      size_t minRangePos = metric.pos % metricInfo.timestamps.size();
      size_t maxRangePos = (size_t)(metric.pos) - 1 % metricInfo.timestamps.size();
      auto &timestamps = metricInfo.timestamps[mi];
      size_t curMinTime = timestamps[minRangePos];
      size_t curMaxTime = timestamps[maxRangePos];
      minTime = minTime < curMinTime ? minTime : curMinTime;
      maxTime = maxTime > curMaxTime ? maxTime : curMaxTime;
    }
    ImGui::Text("min timestamp: %zu, max timestamp: %zu",
                minTime, maxTime);
  }
  ImGui::BeginChild("ScrollingRegion", ImVec2(0,-ImGui::GetItemsLineHeightWithSpacing()), false, ImGuiWindowFlags_HorizontalScrollbar);
  ImGui::Columns(2);
  ImGui::SetColumnOffset(1, 300);
  for (size_t i = 0; i < gState.devices.size(); ++i) {
    DeviceGUIState &guiState = gState.devices[i];
    const DeviceSpec &spec = devices[i];
    const DeviceInfo &info = infos[i];
    const DeviceMetricsInfo &metricsInfo = metricsInfos[i];

    historyBar(guiState, spec, metricsInfo);
  }
  ImGui::Columns(1);
  ImGui::EndChild();
  ImGui::End();
}

// FIXME: return empty function in case we were not built
// with GLFW support.
std::function<void(void)>
getGUIDebugger(const std::vector<DeviceInfo> &infos,
               const std::vector<DeviceSpec> &devices,
               const std::vector<DeviceMetricsInfo> &metricsInfos,
               std::vector<DeviceControl> &controls
               ) {
  gState.selectedMetric = -1;
  // FIXME: this should probaly have a better mapping between our window state and
  gState.devices.resize(infos.size());
  for (size_t i = 0; i < gState.devices.size(); ++i) {
    DeviceGUIState &state = gState.devices[i];
    const DeviceSpec &spec = devices[i];
    const DeviceInfo &info = infos[i];
    state.label = devices[i].id + "(" + std::to_string(info.pid) + ")";
  }

  return [&infos, &devices, &controls, &metricsInfos]() {
    ImGuiStyle &style = ImGui::GetStyle();
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.09f, 0.09f, 1.00f);

    displayDeviceHistograms(infos, devices, controls, metricsInfos);
    int windowPosStepping = (ImGui::GetIO().DisplaySize.y - 500) / gState.devices.size();
    for (size_t i = 0; i < gState.devices.size(); ++i) {
      DeviceGUIState &state = gState.devices[i];
      const DeviceInfo &info = infos[i];
      const DeviceSpec &spec = devices[i];

      DeviceControl &control = controls[i];
      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x/3*2, i*windowPosStepping), ImGuiSetCond_Once);
      ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x/3, ImGui::GetIO().DisplaySize.y - 300), ImGuiSetCond_Once);
      if (!info.active) {
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.9,0,0,1));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(1,0,0,1));
        ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, ImVec4(1,0,0,1));
      }
      ImGui::Begin(state.label.c_str());
      if (ImGui::CollapsingHeader("Channels")) {
        ImGui::Text("# channels: %lu", spec.channels.size());
        ImGui::Columns(2);
        ImGui::TextUnformatted("Name");
        ImGui::NextColumn();
        ImGui::TextUnformatted("Port");
        ImGui::NextColumn();
        for (auto channel : spec.channels) {
          ImGui::TextUnformatted(channel.name.c_str());
          ImGui::NextColumn();
          ImGui::Text("%d", channel.port);
          ImGui::NextColumn();
        }
        ImGui::Columns(1);
      }
      optionsTable(spec, control);
      if (ImGui::CollapsingHeader("Logs", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Stop logging", &control.quiet);
        ImGui::InputText("Log filter", control.logFilter, sizeof(control.logFilter));
        assert(sizeof(control.logStartTrigger) == 256);
        assert(sizeof(control.logStopTrigger) == 256);
        ImGui::InputText("Log start trigger", control.logStartTrigger, sizeof(control.logStartTrigger));
        ImGui::InputText("Log stop trigger", control.logStopTrigger, sizeof(control.logStopTrigger));

        ImGui::Separator();
        ImGui::BeginChild("ScrollingRegion", ImVec2(0,-ImGui::GetItemsLineHeightWithSpacing()), false, ImGuiWindowFlags_HorizontalScrollbar);
        displayHistory(info, control);
        ImGui::EndChild();
      }
      ImGui::End();
      if (!info.active) {
        ImGui::PopStyleColor(3);
      }
    }
  };
}

} // namespace framework
} // namespace o2
