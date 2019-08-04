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
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "DebugGUI/imgui.h"
#include "DebugGUI/imgui_extras.h"
#include "DriverControl.cxx"
#include "DriverInfo.cxx"
#include "FrameworkGUIDeviceInspector.h"
#include "Framework/FrameworkGUIDevicesGraph.h"
#include "Framework/FrameworkGUIDataRelayerUsage.h"
#include "Framework/PaletteHelpers.h"
#include "Framework/FrameworkGUIState.h"

// Simplify debugging
template class std::vector<o2::framework::DeviceMetricsInfo>;

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2
{
namespace framework
{
namespace gui
{
// Type erased information for the plotting
struct MultiplotData {
  int mod;
  size_t first;
  size_t size;
  const void* Y;
  const void* X;
  MetricType type;
};

} // namespace gui
} // namespace framework
} // namespace o2

template class std::vector<o2::framework::gui::MultiplotData>;

namespace o2
{
namespace framework
{
namespace gui
{

ImVec4 colorForLogLevel(LogParsingHelpers::LogLevel logLevel)
{
  switch (logLevel) {
    case LogParsingHelpers::LogLevel::Info:
      return PaletteHelpers::GREEN;
    case LogParsingHelpers::LogLevel::Debug:
      return ImVec4(153. / 255, 61. / 255, 61. / 255, 255. / 255);
    case LogParsingHelpers::LogLevel::Warning:
      return PaletteHelpers::DARK_YELLOW;
    case LogParsingHelpers::LogLevel::Error:
      return PaletteHelpers::RED;
    case LogParsingHelpers::LogLevel::Unknown:
      return ImVec4(153. / 255, 61. / 255, 61. / 255, 255. / 255);
    default:
      return PaletteHelpers::WHITE;
  };
}

void displayHistory(const DeviceInfo& info, DeviceControl& control)
{
  if (info.history.empty()) {
    return;
  }
  int startPos = info.historyPos;
  const int historySize = info.history.size();

  int triggerStartPos = startPos + 1 % historySize;
  int triggerStopPos = startPos % historySize;

  int j = startPos;
  // We look for a stop trigger, so that we know where to stop the search for
  // out start search. If no stop trigger is found, we search until the end
  if (control.logStopTrigger[0]) {
    while ((j % historySize) != ((startPos + 1) % historySize)) {
      assert(j >= 0);
      assert(j < historySize);
      auto& line = info.history[j];
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
    while ((j % historySize) != triggerStopPos) {
      assert(historySize > j);
      assert(historySize == 1000);
      auto& line = info.history[j];
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
  while (historySize && ((ji % historySize) != je)) {
    assert(iterations < historySize);
    iterations++;
    assert(historySize == 1000);
    assert(ji < historySize);
    auto& line = info.history[ji];
    auto logLevel = info.historyLevel[ji];

    // Skip empty lines
    if (line.empty()) {
      ji = (ji + 1) % historySize;
      continue;
    }
    // Print matching lines
    if (strstr(line.c_str(), control.logFilter) != nullptr) {
      auto color = colorForLogLevel(logLevel);
      // We filter twice, once on input, to reduce the
      // stream, a second time at display time, to avoid
      // showing unrelevant messages from past.
      if (logLevel >= control.logLevel) {
        ImGui::TextColored(color, line.c_str(), line.c_str() + line.size());
      }
    }
    ji = (ji + 1) % historySize;
  }
}

template <typename T>
struct HistoData {
  int mod;
  size_t first;
  size_t size;
  const T* points;
};

enum struct MetricsDisplayStyle : int {
  Lines = 0,
  Histos = 1,
  Sparks = 2,
  Table = 3
};

void displayDeviceMetrics(const char* label, ImVec2 canvasSize, std::string const& selectedMetricName,
                          size_t rangeBegin, size_t rangeEnd, size_t bins, MetricsDisplayStyle displayType,
                          std::vector<DeviceSpec> const& specs, std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  static std::vector<ImColor> palette = {
    ImColor{218, 124, 48},
    ImColor{62, 150, 81},
    ImColor{204, 37, 41},
    ImColor{83, 81, 84},
    ImColor{107, 76, 154},
    ImColor{146, 36, 40},
    ImColor{148, 139, 61}};
  std::vector<void const*> metricsToDisplay;
  std::vector<const char*> deviceNames;
  std::vector<MultiplotData> userData;
  std::vector<ImColor> colors;
  MetricType metricType;
  size_t metricSize = 0;
  assert(specs.size() == metricsInfos.size());
  float maxValue = std::numeric_limits<float>::lowest();
  float minValue = 0;
  size_t maxDomain = std::numeric_limits<size_t>::lowest();
  size_t minDomain = std::numeric_limits<size_t>::max();

  for (int mi = 0; mi < metricsInfos.size(); ++mi) {
    auto vi = DeviceMetricsHelper::metricIdxByName(selectedMetricName, metricsInfos[mi]);
    if (vi == metricsInfos[mi].metricLabelsIdx.size()) {
      continue;
    }
    auto& metric = metricsInfos[mi].metrics[vi];
    deviceNames.push_back(specs[mi].name.c_str());
    colors.push_back(palette[mi % palette.size()]);
    metricType = metric.type;
    MultiplotData data;
    data.mod = metricsInfos[mi].timestamps[vi].size();
    data.first = metric.pos - data.mod;
    data.X = metricsInfos[mi].timestamps[vi].data();
    minValue = std::min(minValue, metricsInfos[mi].min[vi]);
    maxValue = std::max(maxValue, metricsInfos[mi].max[vi]);
    minDomain = std::min(minDomain, metricsInfos[mi].minDomain[vi]);
    maxDomain = std::max(maxDomain, metricsInfos[mi].maxDomain[vi]);
    switch (metric.type) {
      case MetricType::Int: {
        data.size = metricsInfos[mi].intMetrics[metric.storeIdx].size();
        data.Y = metricsInfos[mi].intMetrics[metric.storeIdx].data();
        data.type = MetricType::Int;
        metricType = MetricType::Int;
        metricSize = metricsInfos[mi].intMetrics[metric.storeIdx].size();
      } break;
      case MetricType::Float: {
        data.size = metricsInfos[mi].floatMetrics[metric.storeIdx].size();
        data.Y = metricsInfos[mi].floatMetrics[metric.storeIdx].data();
        data.type = MetricType::Float;
        metricType = MetricType::Float;
        metricSize = metricsInfos[mi].floatMetrics[metric.storeIdx].size();
      } break;
      case MetricType::Unknown:
      case MetricType::String: {
        data.size = metricsInfos[mi].stringMetrics[metric.storeIdx].size();
        data.Y = nullptr;
        data.type = MetricType::String;
        metricType = MetricType::String;
        metricSize = metricsInfos[mi].stringMetrics[metric.storeIdx].size();
      } break;
    }
    userData.emplace_back(data);
  }

  maxDomain = std::max(minDomain + 1024, maxDomain);

  for (size_t ui = 0; ui < userData.size(); ++ui) {
    metricsToDisplay.push_back(&(userData[ui]));
  }
  auto getterY = [](const void* hData, int idx) -> float {
    auto histoData = reinterpret_cast<const MultiplotData*>(hData);
    size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
    // size_t pos = (static_cast<size_t>(idx)) % histoData->mod;
    assert(pos >= 0 && pos < 1024);
    if (histoData->type == MetricType::Int) {
      return static_cast<const int*>(histoData->Y)[pos];
    } else if (histoData->type == MetricType::Float) {
      return static_cast<const float*>(histoData->Y)[pos];
    } else {
      return 0;
    }
  };
  auto getterX = [](const void* hData, int idx) -> size_t {
    auto histoData = reinterpret_cast<const MultiplotData*>(hData);
    size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
    //size_t pos = (static_cast<size_t>(idx)) % histoData->mod;
    assert(pos >= 0 && pos < 1024);
    return static_cast<const size_t*>(histoData->X)[pos];
  };
  switch (displayType) {
    case MetricsDisplayStyle::Histos:
      ImGui::PlotMultiHistograms(
        label,
        userData.size(),
        deviceNames.data(),
        colors.data(),
        getterY,
        getterX,
        metricsToDisplay.data(),
        metricSize,
        minValue,
        maxValue * 1.2f,
        minDomain,
        maxDomain,
        canvasSize);
      break;
    case MetricsDisplayStyle::Lines:
      ImGui::PlotMultiLines(
        label,
        userData.size(),
        deviceNames.data(),
        colors.data(),
        getterY,
        getterX,
        metricsToDisplay.data(),
        metricSize,
        minValue,
        maxValue * 1.2f,
        minDomain,
        maxDomain,
        canvasSize);
      break;
    default:
      break;
  }
}

struct ColumnInfo {
  MetricType type;
  int index;
};

void metricsTableRow(std::vector<ColumnInfo> columnInfos,
                     std::vector<DeviceMetricsInfo> const& metricsInfos,
                     int row)
{
  ImGui::Text("%d", row);
  ImGui::NextColumn();

  for (size_t j = 0; j < columnInfos.size(); j++) {
    auto& info = columnInfos[j];
    auto& metricsInfo = metricsInfos[j];

    if (info.index == -1) {
      ImGui::TextUnformatted("-");
      ImGui::NextColumn();
      continue;
    }
    switch (info.type) {
      case MetricType::Int: {
        ImGui::Text("%i (%i)", metricsInfo.intMetrics[info.index][row], info.index);
        ImGui::NextColumn();
      } break;
      case MetricType::Float: {
        ImGui::Text("%f (%i)", metricsInfo.floatMetrics[info.index][row], info.index);
        ImGui::NextColumn();
      } break;
      case MetricType::String: {
        ImGui::Text("%s (%i)", metricsInfo.stringMetrics[info.index][row].data, info.index);
        ImGui::NextColumn();
      } break;
      default:
        ImGui::NextColumn();
        break;
    }
  }
}

void historyBar(gui::WorkspaceGUIState& globalGUIState,
                size_t rangeBegin, size_t rangeEnd,
                gui::DeviceGUIState& state,
                DriverInfo const& driverInfo,
                DeviceSpec const& spec,
                DeviceMetricsInfo const& metricsInfo)
{
  bool open = ImGui::TreeNode(state.label.c_str());
  if (open) {
    ImGui::Text("# channels: %lu", spec.outputChannels.size() + spec.inputChannels.size());
    ImGui::TreePop();
  }
  ImGui::NextColumn();

  if (globalGUIState.selectedMetric == -1) {
    ImGui::NextColumn();
    return;
  }

  auto currentMetricName = driverInfo.availableMetrics[globalGUIState.selectedMetric];

  size_t i = DeviceMetricsHelper::metricIdxByName(currentMetricName, metricsInfo);
  // We did not find any plot, skipping this.
  if (i == metricsInfo.metricLabelsIdx.size()) {
    ImGui::NextColumn();
    return;
  }
  auto& metric = metricsInfo.metrics[i];

  switch (metric.type) {
    case MetricType::Int: {
      HistoData<int> data;
      data.mod = metricsInfo.timestamps[i].size();
      data.first = metric.pos - data.mod;
      data.size = metricsInfo.intMetrics[metric.storeIdx].size();
      data.points = metricsInfo.intMetrics[metric.storeIdx].data();

      auto getter = [](void* hData, int idx) -> float {
        auto histoData = reinterpret_cast<HistoData<int>*>(hData);
        size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
        assert(pos >= 0 && pos < 1024);
        return histoData->points[pos];
      };
      ImGui::PlotLines(("##" + currentMetricName).c_str(), getter, &data, data.size);
      ImGui::NextColumn();
    } break;
    case MetricType::Float: {
      HistoData<float> data;
      data.mod = metricsInfo.timestamps[i].size();
      data.first = metric.pos - data.mod;
      data.size = metricsInfo.floatMetrics[metric.storeIdx].size();
      data.points = metricsInfo.floatMetrics[metric.storeIdx].data();

      auto getter = [](void* hData, int idx) -> float {
        auto histoData = reinterpret_cast<HistoData<float>*>(hData);
        size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
        assert(pos >= 0 && pos < 1024);
        return histoData->points[pos];
      };
      ImGui::PlotLines(("##" + currentMetricName).c_str(), getter, &data, data.size);
      ImGui::NextColumn();
    } break;
    default:
      ImGui::NextColumn();
      return;
      break;
  }
}

/// Calculate where to find the coliumns for a give metric
std::vector<ColumnInfo> calculateTableIndex(gui::WorkspaceGUIState& globalGUIState,
                                            int selectedMetric,
                                            DriverInfo const& driverInfo,
                                            std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  std::vector<ColumnInfo> columns;
  for (size_t j = 0; j < globalGUIState.devices.size(); ++j) {
    const DeviceMetricsInfo& metricsInfo = metricsInfos[j];
    /// Nothing to draw, if no metric selected.
    if (selectedMetric == -1) {
      columns.push_back({MetricType::Int, -1});
      continue;
    }
    auto currentMetricName = driverInfo.availableMetrics[selectedMetric];
    size_t idx = DeviceMetricsHelper::metricIdxByName(currentMetricName, metricsInfo);

    // We did not find any plot, skipping this.
    if (idx == metricsInfo.metricLabelsIdx.size()) {
      columns.push_back({MetricType::Int, -1});
      continue;
    }
    auto metric = metricsInfos[j].metrics[idx];
    columns.push_back({metric.type, static_cast<int>(metric.storeIdx)});
  }
  return columns;
};

void displayDeviceHistograms(gui::WorkspaceGUIState& state,
                             DriverInfo const& driverInfo,
                             std::vector<DeviceInfo> const& infos,
                             std::vector<DeviceSpec> const& devices,
                             std::vector<DataProcessorInfo> const& metadata,
                             std::vector<DeviceControl>& controls,
                             std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  showTopologyNodeGraph(state, infos, devices, metadata, controls, metricsInfos);
  if (state.bottomPaneVisible == false) {
    return;
  }
  ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - state.bottomPaneSize), 0);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, state.bottomPaneSize), 0);

  ImGui::Begin("Devices", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
  ImGui::BeginGroup();
  char const* currentMetric = nullptr;
  if (state.selectedMetric != -1) {
    currentMetric = driverInfo.availableMetrics[state.selectedMetric].c_str();
  } else {
    currentMetric = "Click to select metric";
  }
  if (ImGui::BeginCombo("###Select metric", currentMetric, 0)) {
    for (size_t mi = 0; mi < driverInfo.availableMetrics.size(); ++mi) {
      auto metric = driverInfo.availableMetrics[mi];
      bool isSelected = mi == state.selectedMetric;
      if (ImGui::Selectable(driverInfo.availableMetrics[mi].c_str(), isSelected)) {
        state.selectedMetric = mi;
      }
      if (isSelected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  };

  static char const* plotStyles[] = {
    "lines",
    "histograms",
    "sparks",
    "table"};
  ImGui::SameLine();
  static enum MetricsDisplayStyle currentStyle = MetricsDisplayStyle::Lines;
  ImGui::Combo("##Select style", reinterpret_cast<int*>(&currentStyle), plotStyles, IM_ARRAYSIZE(plotStyles));

  // Calculate the full timestamp range for the selected metric
  size_t minTime = -1;
  size_t maxTime = 0;
  std::string currentMetricName;
  if (state.selectedMetric >= 0) {
    currentMetricName = driverInfo.availableMetrics[state.selectedMetric];
    for (auto& metricInfo : metricsInfos) {
      size_t mi = DeviceMetricsHelper::metricIdxByName(currentMetricName, metricInfo);
      if (mi == metricInfo.metricLabelsIdx.size()) {
        continue;
      }
      auto& metric = metricInfo.metrics[mi];
      auto& timestamps = metricInfo.timestamps[mi];

      for (size_t ti = 0; ti != metricInfo.timestamps.size(); ++ti) {
        size_t minRangePos = (metric.pos + ti) % metricInfo.timestamps.size();
        size_t curMinTime = timestamps[minRangePos];
        if (curMinTime == 0) {
          continue;
        }
        minTime = minTime < curMinTime ? minTime : curMinTime;
        if (minTime != 0 && minTime != -1) {
          break;
        }
      }
      size_t maxRangePos = (size_t)(metric.pos) - 1 % metricInfo.timestamps.size();
      size_t curMaxTime = timestamps[maxRangePos];
      maxTime = maxTime > curMaxTime ? maxTime : curMaxTime;
    }
  }
  if (minTime != -1) {
    ImGui::Text("min timestamp: %zu, max timestamp: %zu", minTime, maxTime);
  }
  ImGui::EndGroup();
  if (!currentMetricName.empty()) {
    switch (currentStyle) {
      case MetricsDisplayStyle::Histos:
      case MetricsDisplayStyle::Lines: {
        displayDeviceMetrics("Metrics",
                             ImVec2(ImGui::GetIO().DisplaySize.x - 10, state.bottomPaneSize - ImGui::GetItemRectSize().y - 20), currentMetricName, minTime, maxTime, 1024,
                             currentStyle, devices, metricsInfos);
      } break;
      case MetricsDisplayStyle::Sparks: {
        ImGui::BeginChild("##ScrollingRegion", ImVec2(ImGui::GetIO().DisplaySize.x + state.leftPaneSize + state.rightPaneSize - 10, -ImGui::GetItemsLineHeightWithSpacing()), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::Columns(2);
        ImGui::SetColumnOffset(1, 300);
        for (size_t i = 0; i < state.devices.size(); ++i) {
          gui::DeviceGUIState& deviceGUIState = state.devices[i];
          const DeviceSpec& spec = devices[i];
          const DeviceMetricsInfo& metricsInfo = metricsInfos[i];

          historyBar(state, minTime, maxTime, deviceGUIState, driverInfo, spec, metricsInfo);
        }
        ImGui::Columns(1);
        ImGui::EndChild();
      } break;
      case MetricsDisplayStyle::Table: {
        ImGui::BeginChild("##ScrollingRegion", ImVec2(ImGui::GetIO().DisplaySize.x + state.leftPaneSize + state.rightPaneSize - 10, -ImGui::GetItemsLineHeightWithSpacing()), false,
                          ImGuiWindowFlags_HorizontalScrollbar);

        // The +1 is for the timestamp column
        ImGui::Columns(state.devices.size() + 1);
        ImGui::TextUnformatted("entry");
        ImGui::NextColumn();
        ImVec2 textsize = ImGui::CalcTextSize("extry", nullptr, true);
        float offset = 0.f;
        offset += std::max(100.f, textsize.x);
        for (size_t j = 0; j < state.devices.size(); ++j) {
          gui::DeviceGUIState& deviceGUIState = state.devices[j];
          const DeviceSpec& spec = devices[j];

          ImGui::SetColumnOffset(-1, offset);
          textsize = ImGui::CalcTextSize(spec.name.c_str(), nullptr, true);
          offset += std::max(100.f, textsize.x);
          ImGui::TextUnformatted(spec.name.c_str());
          ImGui::NextColumn();
        }
        ImGui::Separator();

        auto columns = calculateTableIndex(state, state.selectedMetric, driverInfo, metricsInfos);

        // Calculate which columns we want to see.
        // FIXME: only one column for now.
        for (size_t i = 0; i < 10; ++i) {
          metricsTableRow(columns, metricsInfos, i);
        }
        ImGui::Columns(1);

        ImGui::EndChild();
      } break;
    }
  }
  ImGui::End();
}

void pushWindowColorDueToStatus(const DeviceInfo& info)
{
  using LogLevel = LogParsingHelpers::LogLevel;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.);
  if (info.active == false) {
    ImGui::PushStyleColor(ImGuiCol_TitleBg, PaletteHelpers::DARK_RED);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, PaletteHelpers::RED);
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, PaletteHelpers::RED);
    return;
  }
  switch (info.maxLogLevel) {
    case LogLevel::Error:
      ImGui::PushStyleColor(ImGuiCol_TitleBg, PaletteHelpers::SHADED_RED);
      ImGui::PushStyleColor(ImGuiCol_TitleBgActive, PaletteHelpers::RED);
      ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, PaletteHelpers::SHADED_RED);
      break;
    case LogLevel::Warning:
      ImGui::PushStyleColor(ImGuiCol_TitleBg, PaletteHelpers::SHADED_YELLOW);
      ImGui::PushStyleColor(ImGuiCol_TitleBgActive, PaletteHelpers::YELLOW);
      ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, PaletteHelpers::SHADED_YELLOW);
      break;
    case LogLevel::Info:
      ImGui::PushStyleColor(ImGuiCol_TitleBg, PaletteHelpers::SHADED_GREEN);
      ImGui::PushStyleColor(ImGuiCol_TitleBgActive, PaletteHelpers::GREEN);
      ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, PaletteHelpers::SHADED_GREEN);
      break;
    default:
      ImGui::PushStyleColor(ImGuiCol_TitleBg, PaletteHelpers::SHADED_BLUE);
      ImGui::PushStyleColor(ImGuiCol_TitleBgActive, PaletteHelpers::BLUE);
      ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, PaletteHelpers::SHADED_BLUE);
      break;
  }
}

void popWindowColorDueToStatus()
{
  ImGui::PopStyleColor(3);
  ImGui::PopStyleVar(1);
}

struct DriverHelper {
  static char const* stateToString(enum DriverState state)
  {
    static const char* names[static_cast<int>(DriverState::LAST)] = {
      "INIT",             //
      "SCHEDULE",         //
      "RUNNING",          //
      "GUI",              //
      "REDEPLOY_GUI",     //
      "QUIT_REQUESTED",   //
      "HANDLE_CHILDREN",  //
      "EXIT",             //
      "UNKNOWN"           //
      "PERFORM_CALLBACKS" //
    };
    return names[static_cast<int>(state)];
  }
};

/// Display information window about the driver
/// and its state.
void displayDriverInfo(DriverInfo const& driverInfo, DriverControl& driverControl)
{
  ImGui::Begin("Driver information");
  ImGui::Text("Numer of running devices: %lu", driverInfo.socket2DeviceInfo.size() / 2);

  if (driverControl.state == DriverControlState::STEP) {
    driverControl.state = DriverControlState::PAUSE;
  }
  auto state = reinterpret_cast<int*>(&driverControl.state);
  ImGui::RadioButton("Play", state, static_cast<int>(DriverControlState::PLAY));
  ImGui::SameLine();
  ImGui::RadioButton("Pause", state, static_cast<int>(DriverControlState::PAUSE));
  ImGui::SameLine();
  ImGui::RadioButton("Step", state, static_cast<int>(DriverControlState::STEP));

  if (driverControl.state == DriverControlState::PAUSE) {
    driverControl.forcedTransitions.push_back(DriverState::GUI);
  }

  auto& registry = driverInfo.configContext->options();
  ImGui::Columns();

  ImGui::Text("Frame cost (latency): %.1f(%.1f)ms", driverInfo.frameCost, driverInfo.frameLatency);
  ImGui::Text("Input parsing cost (latency): %.1f(%.1f)ms", driverInfo.inputProcessingCost, driverInfo.inputProcessingLatency);
  ImGui::Text("State stack (depth %lu)", driverInfo.states.size());
  if (ImGui::Button("SIGCONT all children")) {
    kill(0, SIGCONT);
  }

  for (size_t i = 0; i < driverInfo.states.size(); ++i) {
    ImGui::Text("#%lu: %s", i, DriverHelper::stateToString(driverInfo.states[i]));
  }

  ImGui::End();
}

// FIXME: return empty function in case we were not built
// with GLFW support.
///
std::function<void(void)> getGUIDebugger(std::vector<DeviceInfo> const& infos,
                                         std::vector<DeviceSpec> const& devices,
                                         std::vector<DataProcessorInfo> const& metadata,
                                         std::vector<DeviceMetricsInfo> const& metricsInfos,
                                         DriverInfo const& driverInfo,
                                         std::vector<DeviceControl>& controls,
                                         DriverControl& driverControl)
{
  static gui::WorkspaceGUIState globalGUIState;
  gui::WorkspaceGUIState& guiState = globalGUIState;
  guiState.selectedMetric = -1;
  guiState.metricMaxRange = 0UL;
  guiState.metricMinRange = -1;
  // FIXME: this should probaly have a better mapping between our window state and
  guiState.devices.resize(infos.size());
  for (size_t i = 0; i < guiState.devices.size(); ++i) {
    gui::DeviceGUIState& state = guiState.devices[i];
    state.label = devices[i].id + "(" + std::to_string(infos[i].pid) + ")";
  }
  guiState.bottomPaneSize = 300;
  guiState.leftPaneSize = 200;
  guiState.rightPaneSize = 300;

  // Show all the panes by default.
  guiState.bottomPaneVisible = true;
  guiState.leftPaneVisible = true;
  guiState.rightPaneVisible = true;

  return [&guiState, &infos, &devices, &metadata, &controls, &metricsInfos, &driverInfo, &driverControl]() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 0.;
    style.WindowRounding = 0.;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0x1b / 255.f, 0x1b / 255.f, 0x1b / 255.f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0x1b / 255.f, 0x1b / 255.f, 0x1b / 255.f, 1.00f);

    displayDeviceHistograms(guiState, driverInfo, infos, devices, metadata, controls, metricsInfos);
    displayDriverInfo(driverInfo, driverControl);

    int windowPosStepping = (ImGui::GetIO().DisplaySize.y - 500) / guiState.devices.size();

    for (size_t i = 0; i < guiState.devices.size(); ++i) {
      gui::DeviceGUIState& state = guiState.devices[i];
      assert(i < infos.size());
      assert(i < devices.size());
      const DeviceInfo& info = infos[i];
      const DeviceSpec& spec = devices[i];
      const DeviceMetricsInfo& metrics = metricsInfos[i];

      assert(controls.size() == devices.size());
      DeviceControl& control = controls[i];

      pushWindowColorDueToStatus(info);
      if (control.logVisible) {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 3 * 2, i * windowPosStepping), ImGuiSetCond_Once);
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x / 3, ImGui::GetIO().DisplaySize.y - 300),
                                 ImGuiSetCond_Once);
        ImGui::Begin(state.label.c_str(), &control.logVisible);

        ImGui::InputText("Log filter", control.logFilter, sizeof(control.logFilter));
        ImGui::InputText("Log start trigger", control.logStartTrigger, sizeof(control.logStartTrigger));
        ImGui::InputText("Log stop trigger", control.logStopTrigger, sizeof(control.logStopTrigger));
        ImGui::Checkbox("Stop logging", &control.quiet);
        ImGui::SameLine();
        ImGui::Combo("Log level", reinterpret_cast<int*>(&control.logLevel), LogParsingHelpers::LOG_LEVELS,
                     (int)LogParsingHelpers::LogLevel::Size, 5);

        ImGui::Separator();
        ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetItemsLineHeightWithSpacing()), false,
                          ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
        displayHistory(info, control);
        ImGui::EndChild();
        ImGui::End();
      }
      popWindowColorDueToStatus();
    }
  };
}

} // namespace gui
} // namespace framework
} // namespace o2
