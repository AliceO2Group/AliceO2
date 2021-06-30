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
#include "FrameworkGUIDebugger.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "DebugGUI/imgui.h"
#include "DebugGUI/implot.h"
#include "DebugGUI/imgui_extras.h"
#include "Framework/DriverControl.h"
#include "Framework/DriverInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include "FrameworkGUIDeviceInspector.h"
#include "FrameworkGUIDevicesGraph.h"
#include "FrameworkGUIDataRelayerUsage.h"
#include "PaletteHelpers.h"
#include "FrameworkGUIState.h"

#include <fmt/format.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <cinttypes>

// Simplify debugging
template class std::vector<o2::framework::DeviceMetricsInfo>;

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2::framework::gui
{
// Type erased information for the plotting
struct MultiplotData {
  int mod;
  size_t first;
  size_t size;
  const void* Y = nullptr;
  const void* X = nullptr;
  MetricType type;
  const char* legend = nullptr;
  int axis = 0;
};

} // namespace o2::framework::gui

template class std::vector<o2::framework::gui::MultiplotData>;

namespace o2::framework::gui
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
      return ImVec4(194. / 255, 195. / 255, 199. / 255, 255. / 255);
    default:
      return ImVec4(194. / 255, 195. / 255, 199. / 255, 255. / 255);
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
        if (line.find('%', 0) != std::string::npos) {
          ImGui::TextUnformatted(line.c_str(), line.c_str() + line.size());
        } else {
          ImGui::TextColored(color, line.c_str(), line.c_str() + line.size());
        }
      }
    }
    ji = (ji + 1) % historySize;
  }
}

struct HistoData {
  int mod;
  size_t first;
  size_t size;
  void* points = nullptr;
  const size_t* time = nullptr;
  char const* legend = nullptr;
};

enum struct MetricsDisplayStyle : int {
  Lines = 0,
  Histos = 1,
  Sparks = 2,
  Table = 3,
  Stems = 4
};

/// Information associated to a node in the topology
struct TopologyNodeInfo {
  std::string label;
};

struct MetricDisplayState {
  int axis = 0; // The Axis to use for Y
  bool visible = false;
  bool selected = false;
  std::string legend;
  size_t legendHash = -1;
};

struct MetricIndex {
  size_t storeIndex;
  size_t deviceIndex;
  size_t metricIndex;
  size_t stateIndex;
};

enum MetricTypes {
  DEVICE_METRICS = 0,
  DRIVER_METRICS,
  TOTAL_TYPES_OF_METRICS
};

// We use this to keep together all the kind of metrics
// so that we can display driver and device metrics in the same plot
// without an if.
struct AllMetricsStore {
  std::vector<DeviceMetricsInfo> const* metrics[TOTAL_TYPES_OF_METRICS];
  std::vector<TopologyNodeInfo> const* specs[TOTAL_TYPES_OF_METRICS];
};

void displaySparks(
  double startTime,
  std::vector<MetricIndex>& visibleMetricsIndex,
  std::vector<MetricDisplayState>& metricDisplayStates,
  AllMetricsStore const& metricStore)
{
  static bool locked = false;
  ImGui::Checkbox("Lock scrolling", &locked);
  ImGui::BeginTable("##sparks table", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY, ImVec2{-1, -1});
  ImGui::TableSetupColumn("##close button", ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_WidthFixed, 25);
  ImGui::TableSetupColumn("##plot name", ImGuiTableColumnFlags_WidthFixed, 200);
  ImGui::TableSetupColumn("##plot", ImGuiTableColumnFlags_WidthFixed, 0);
  ImGui::TableSetupScrollFreeze(2, 0);
  for (size_t i = 0; i < visibleMetricsIndex.size(); ++i) {
    auto& index = visibleMetricsIndex[i];
    auto& metricsInfos = *metricStore.metrics[index.storeIndex];
    auto& nodes = *metricStore.specs[index.storeIndex];
    auto& metricsInfo = metricsInfos[index.deviceIndex];
    auto& label = metricsInfo.metricLabels[index.metricIndex];
    auto& metric = metricsInfo.metrics[index.metricIndex];
    auto& state = metricDisplayStates[index.stateIndex];

    ImGui::TableNextColumn();
    ImGui::PushID(index.stateIndex);
    state.visible = !ImGui::Button("-");
    ImGui::PopID();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(state.legend.c_str());
    ImGui::TableNextColumn();
    static ImPlotAxisFlags rtx_axis = ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickMarks;
    static ImPlotAxisFlags rty_axis = ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickMarks;
    ImGui::PushID(index.stateIndex);
    HistoData data;
    data.mod = std::min(metric.filledMetrics, metricsInfo.timestamps[index.metricIndex].size());
    data.first = metric.pos - data.mod;
    data.size = metric.filledMetrics;
    data.time = metricsInfo.timestamps[index.metricIndex].data();
    data.legend = state.legend.c_str();

    if (!locked) {
      ImPlot::SetNextPlotLimitsX((startTime + ImGui::GetTime() - 100) * 1000, (startTime + ImGui::GetTime()) * 1000, ImGuiCond_Always);
      ImPlot::SetNextPlotLimitsY(metricsInfo.min[index.metricIndex], metricsInfo.max[index.metricIndex] * 1.1, ImGuiCond_Always);
      rty_axis |= ImPlotAxisFlags_LockMin;
    }
    if (ImPlot::BeginPlot("##sparks", "time", "value", ImVec2(700, 100), 0, rtx_axis, rty_axis)) {
      ImPlot::SetPlotYAxis(state.axis);
      switch (metric.type) {
        case MetricType::Int: {
          data.points = (void*)metricsInfo.intMetrics[metric.storeIdx].data();

          auto getter = [](void* hData, int idx) -> ImPlotPoint {
            auto histoData = reinterpret_cast<HistoData*>(hData);
            size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
            assert(pos >= 0 && pos < 1024);
            return ImPlotPoint(histoData->time[pos], ((int*)(histoData->points))[pos]);
          };
          ImPlot::PlotLineG("##plot", getter, &data, data.size);
        } break;
        case MetricType::Uint64: {
          data.points = (void*)metricsInfo.uint64Metrics[metric.storeIdx].data();

          auto getter = [](void* hData, int idx) -> ImPlotPoint {
            auto histoData = reinterpret_cast<HistoData*>(hData);
            size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
            assert(pos >= 0 && pos < 1024);
            return ImPlotPoint(histoData->time[pos], ((uint64_t*)histoData->points)[pos]);
          };
          ImPlot::PlotLineG("##plot", getter, &data, data.size, 0);
        } break;
        case MetricType::Float: {
          data.points = (void*)metricsInfo.floatMetrics[metric.storeIdx].data();

          auto getter = [](void* hData, int idx) -> ImPlotPoint {
            auto histoData = reinterpret_cast<HistoData*>(hData);
            size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
            assert(pos >= 0 && pos < 1024);
            return ImPlotPoint(histoData->time[pos], ((float*)histoData->points)[pos]);
          };
          ImPlot::PlotLineG("##plot", getter, &data, data.size, 0);
        } break;
        default:
          return;
          break;
      }
      ImPlot::EndPlot();
    }
    ImGui::PopID();
  }
  ImGui::EndTable();
}

void displayDeviceMetrics(const char* label,
                          size_t rangeBegin, size_t rangeEnd, size_t bins, MetricsDisplayStyle displayType,
                          std::vector<MetricDisplayState>& state,
                          AllMetricsStore const& metricStore)
{
  std::vector<void*> metricsToDisplay;
  std::vector<const char*> deviceNames;
  std::vector<MultiplotData> userData;
  MetricType metricType;
#ifdef NDEBUG
  for (size_t si = 0; si < TOTAL_TYPES_OF_METRICS; ++si) {
    assert(metricsStore.metrics[si].size() == metricStore.specs[si].size());
  }
#endif
  float maxValue[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
  float minValue[3] = {0, 0, 0};
  size_t maxDomain = std::numeric_limits<size_t>::lowest();
  size_t minDomain = std::numeric_limits<size_t>::max();
  size_t gmi = 0;

  ImPlotFlags axisFlags = 0;

  for (size_t si = 0; si < TOTAL_TYPES_OF_METRICS; ++si) {
    std::vector<DeviceMetricsInfo> const& metricsInfos = *metricStore.metrics[si];
    auto const& specs = *metricStore.specs[si];
    for (int di = 0; di < metricsInfos.size(); ++di) {
      for (size_t mi = 0; mi < metricsInfos[di].metrics.size(); ++mi) {
        auto& label = metricsInfos[di].metricLabels[mi];
        if (state[gmi].visible == false) {
          gmi++;
          continue;
        }
        auto& metric = metricsInfos[di].metrics[mi];
        deviceNames.push_back(specs[di].label.c_str());
        metricType = metric.type;
        MultiplotData data;
        data.size = metric.filledMetrics;
        data.mod = std::min(metric.filledMetrics, metricsInfos[di].timestamps[mi].size());
        data.first = metric.pos - data.mod;
        data.X = metricsInfos[di].timestamps[mi].data();
        data.legend = state[gmi].legend.c_str();
        data.type = metric.type;
        data.axis = state[gmi].axis;
        minValue[data.axis] = std::min(minValue[data.axis], metricsInfos[di].min[mi]);
        maxValue[data.axis] = std::max(maxValue[data.axis], metricsInfos[di].max[mi]);
        minDomain = std::min(minDomain, metricsInfos[di].minDomain[mi]);
        maxDomain = std::max(maxDomain, metricsInfos[di].maxDomain[mi]);
        axisFlags |= data.axis == 1 ? ImPlotFlags_YAxis2 : ImPlotFlags_None;
        axisFlags |= data.axis == 2 ? ImPlotFlags_YAxis3 : ImPlotFlags_None;
        switch (metric.type) {
          case MetricType::Int: {
            data.Y = metricsInfos[di].intMetrics[metric.storeIdx].data();
          } break;
          case MetricType::Uint64: {
            data.Y = metricsInfos[di].uint64Metrics[metric.storeIdx].data();
          } break;
          case MetricType::Float: {
            data.Y = metricsInfos[di].floatMetrics[metric.storeIdx].data();
          } break;
          case MetricType::Unknown:
          case MetricType::String: {
            data.Y = nullptr;
            data.type = MetricType::String;
            metricType = MetricType::String;
          } break;
        }

        userData.emplace_back(data);
        gmi++;
      }
    }
  }

  maxDomain = std::max(minDomain + 1024, maxDomain);
  for (size_t ai = 0; ai < 3; ++ai) {
    maxValue[ai] = std::max(minValue[ai] + 1.f, maxValue[ai]);
  }

  static size_t lastMinRange = minDomain;
  static size_t lastMaxRange = maxDomain;

  // Nothing to show.
  if (userData.empty()) {
    return;
  }
  for (size_t ui = 0; ui < userData.size(); ++ui) {
    metricsToDisplay.push_back(&(userData[ui]));
  }

  auto getterXY = [](void* hData, int idx) -> ImPlotPoint {
    auto histoData = reinterpret_cast<const MultiplotData*>(hData);
    size_t pos = (histoData->first + static_cast<size_t>(idx)) % histoData->mod;
    double x = static_cast<const size_t*>(histoData->X)[pos];
    double y = 0.;
    if (histoData->type == MetricType::Int) {
      y = static_cast<const int*>(histoData->Y)[pos];
    } else if (histoData->type == MetricType::Uint64) {
      y = static_cast<const uint64_t*>(histoData->Y)[pos];
    } else if (histoData->type == MetricType::Float) {
      y = static_cast<const float*>(histoData->Y)[pos];
    }
    auto point = ImPlotPoint{x, y};
    return point;
  };
  static bool logScale = false;
  ImGui::Checkbox("Log scale", &logScale);

  ImPlot::SetNextPlotLimitsX(minDomain, maxDomain, ImGuiCond_Once);
  ImPlot::SetNextPlotTicksX(minDomain, maxDomain, 5);
  auto axisPadding = 0.;
  if (displayType == MetricsDisplayStyle::Lines) {
    axisPadding = 0.2;
  }

  for (size_t ai = 0; ai < 3; ++ai) {
    ImPlot::SetNextPlotLimitsY(minValue[ai] - (maxValue[ai] - minValue[ai]) * axisPadding,
                               maxValue[ai] * (1. + axisPadding), ImGuiCond_Always, ai);
  }

  switch (displayType) {
    case MetricsDisplayStyle::Histos:
      if (ImPlot::BeginPlot("##Some plot", "time", "value")) {
        for (size_t pi = 0; pi < metricsToDisplay.size(); ++pi) {
          ImGui::PushID(pi);
          auto data = (const MultiplotData*)metricsToDisplay[pi];
          const char* label = ((MultiplotData*)metricsToDisplay[pi])->legend;
          ImPlot::PlotBarsG(label, getterXY, metricsToDisplay[pi], data->size, 1, 0);
          ImGui::PopID();
        }
        ImPlot::EndPlot();
      }

      break;
    case MetricsDisplayStyle::Lines: {
      auto xAxisFlags = ImPlotAxisFlags_None;
      auto yAxisFlags = ImPlotAxisFlags_LockMin;
      //ImPlot::FitNextPlotAxes(true, true, true, true);
      if (ImPlot::BeginPlot("##Some plot", "time", "value", {-1, -1}, axisFlags, xAxisFlags, yAxisFlags)) {
        for (size_t pi = 0; pi < metricsToDisplay.size(); ++pi) {
          ImGui::PushID(pi);
          auto data = (const MultiplotData*)metricsToDisplay[pi];
          const char* label = data->legend;
          ImPlot::SetPlotYAxis(data->axis);
          ImPlot::PlotLineG(data->legend, getterXY, metricsToDisplay[pi], data->size, 0);
          ImGui::PopID();
        }
        ImPlot::EndPlot();
      }
    } break;
    case MetricsDisplayStyle::Stems:
      if (ImPlot::BeginPlot("##Some plot", "time", "value")) {
        for (size_t pi = 0; pi < userData.size(); ++pi) {
          auto data = reinterpret_cast<const MultiplotData*>(metricsToDisplay[pi]);
          // FIXME: display a message for other metrics
          if (data->type == MetricType::Uint64) {
            ImGui::PushID(pi);
            ImPlot::PlotScatterG(((MultiplotData*)metricsToDisplay[pi])->legend, getterXY, metricsToDisplay[pi], data->size, 0);
            ImGui::PopID();
          }
        }
        ImPlot::EndPlot();
      }
      break;
    default:
      break;
  }
}

struct ColumnInfo {
  MetricType type;
  int index;
};

void metricsTableRow(std::vector<MetricIndex> metricIndex,
                     AllMetricsStore const& metricsStore,
                     int row)
{
  ImGui::TableNextColumn();
  ImGui::Text("%d", row);

  for (auto index : metricIndex) {
    auto& metricsInfos = *metricsStore.metrics[index.storeIndex];
    auto& metricsInfo = metricsInfos[index.deviceIndex];
    auto& info = metricsInfos[index.deviceIndex].metrics[index.metricIndex];

    ImGui::TableNextColumn();
    auto time = metricsInfo.timestamps[index.metricIndex][row];
    switch (info.type) {
      case MetricType::Int: {
        ImGui::Text("%i, %" PRIu64, metricsInfo.intMetrics[info.storeIdx][row], (uint64_t)time);
      } break;
      case MetricType::Uint64: {
        ImGui::Text("%" PRIu64 ", %" PRIu64, metricsInfo.uint64Metrics[info.storeIdx][row], (uint64_t)time);
      } break;
      case MetricType::Float: {
        ImGui::Text("%f, %" PRIu64, metricsInfo.floatMetrics[info.storeIdx][row], (uint64_t)time);
      } break;
      case MetricType::String: {
        ImGui::Text("%s, %" PRIu64, metricsInfo.stringMetrics[info.storeIdx][row].data, (uint64_t)time);
      } break;
      default:
        break;
    }
  }
}

bool hasAll(const char* s, const char* q)
{
  /* base case: empty query */
  if (*q == 0) {
    return true;
  }
  do {
    s = strchr(s, (int)*q);
    if (s == nullptr) {
      return false;
    }
    s++;
    q++;
  } while ((int)*q != 0);
  return true;
}

void TextCenter(char const* text)
{
  float font_size = ImGui::GetFontSize() * strlen(text) / 2;
  ImGui::Dummy(ImVec2(
    ImGui::GetWindowSize().x / 2 -
      font_size + (font_size / 2),
    120));

  ImGui::TextUnformatted(text);
}

void displayMetrics(gui::WorkspaceGUIState& state,
                    DriverInfo const& driverInfo,
                    std::vector<DeviceInfo> const& infos,
                    std::vector<DataProcessorInfo> const& metadata,
                    std::vector<DeviceControl>& controls,
                    AllMetricsStore const& metricsStore)
{
  if (state.bottomPaneVisible == false) {
    return;
  }
  auto metricDisplayPos = 0;
  static bool metricSelectorVisible = true;
  static std::vector<MetricDisplayState> metricDisplayState;
  static bool showInternalMetrics = false;

  // Calculate the full timestamp range for the selected metric
  size_t minTime = -1;
  size_t maxTime = 0;
  constexpr size_t MAX_QUERY_SIZE = 256;
  static char query[MAX_QUERY_SIZE];
  static char lastSelectedQuery[MAX_QUERY_SIZE];

  size_t totalMetrics = 0;
  for (auto& metricsInfos : metricsStore.metrics) {
    for (auto& metricInfo : *metricsInfos) {
      totalMetrics += metricInfo.metrics.size();
    }
  }

  if (totalMetrics != metricDisplayState.size() || strcmp(lastSelectedQuery, query)) {
    size_t gmi = 0;
    std::vector<MetricDisplayState> newMetricDisplayStates;
    newMetricDisplayStates.resize(totalMetrics);
    for (size_t si = 0; si < TOTAL_TYPES_OF_METRICS; ++si) {
      auto& metricsInfos = *metricsStore.metrics[si];
      auto& specs = *metricsStore.specs[si];
      for (size_t di = 0; di < metricsInfos.size(); ++di) {
        auto& metricInfo = metricsInfos[di];
        auto& spec = specs[di];
        for (size_t li = 0; li != metricInfo.metricLabels.size(); ++li) {
          char const* metricLabel = metricInfo.metricLabels[li].label;
          std::string legend = fmt::format("{}/{}", spec.label, metricLabel);
          auto hasher = std::hash<std::string>();
          size_t legendHash = hasher(legend);
          auto old = std::find_if(metricDisplayState.begin(), metricDisplayState.end(), [&legend, &legendHash](MetricDisplayState const& state) { return state.legendHash == legendHash && state.legend == legend; });
          if (old != metricDisplayState.end()) {
            newMetricDisplayStates[gmi].visible = old->visible;
            newMetricDisplayStates[gmi].axis = old->axis;
          } else {
            newMetricDisplayStates[gmi].visible = false;
          }

          newMetricDisplayStates[gmi].selected = hasAll(metricLabel, query);
          newMetricDisplayStates[gmi].legend = legend;
          newMetricDisplayStates[gmi].legendHash = legendHash;
          gmi++;
        }
      }
    }
    metricDisplayState.swap(newMetricDisplayStates);
    strcpy(lastSelectedQuery, query);
  }

  static std::vector<MetricIndex> selectedMetricIndex;

  if (metricSelectorVisible) {
    selectedMetricIndex.clear();
    size_t gmi = 0;
    for (size_t si = 0; si < TOTAL_TYPES_OF_METRICS; ++si) {
      auto& metricsInfos = *metricsStore.metrics[si];
      auto& devices = metricsStore.specs[si];

      for (size_t di = 0; di < metricsInfos.size(); ++di) {
        auto& metricInfo = metricsInfos[di];
        auto& deviceSpec = devices[di];
        for (size_t li = 0; li != metricInfo.metricLabels.size(); ++li) {
          auto& label = metricInfo.metricLabels[li];
          auto& state = metricDisplayState[gmi];
          if (state.selected) {
            selectedMetricIndex.push_back(MetricIndex{si, di, li, gmi});
          }
          gmi++;
        }
      }
    }

    metricDisplayPos = ImGui::GetIO().DisplaySize.x / 4;
    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - state.bottomPaneSize), 0);
    ImGui::SetNextWindowSize(ImVec2(metricDisplayPos, state.bottomPaneSize), 0);
    ImGui::Begin("Available metrics", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    ImGui::Text("Find metrics: ");
    ImGui::SameLine();
    ImGui::InputText("##query-metrics", query, MAX_QUERY_SIZE);
    static const char* possibleAxis[] = {
      "Y",
      "Y1",
      "Y2",
    };
    if (ImGui::BeginTable("##metrics-table", 3, ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY, ImVec2{-1, -1})) {
      ImGui::TableSetupColumn("##close button", ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_WidthFixed, 20);
      ImGui::TableSetupColumn("##axis kind", ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_WidthFixed, 30);
      ImGui::TableSetupScrollFreeze(1, 0);
      ImGuiListClipper clipper;
      clipper.Begin(selectedMetricIndex.size());
      while (clipper.Step()) {
        for (size_t i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
          auto& index = selectedMetricIndex[i];
          auto& metricsInfos = *metricsStore.metrics[index.storeIndex];
          auto& nodes = *metricsStore.specs[index.storeIndex];
          auto& metricInfo = metricsInfos[index.deviceIndex];
          auto& node = nodes[index.deviceIndex];
          auto& label = metricInfo.metricLabels[index.metricIndex];
          auto& metric = metricInfo.metrics[index.metricIndex];
          auto& state = metricDisplayState[index.stateIndex];
          ImGui::PushID(index.stateIndex);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Checkbox("##checkbox", &metricDisplayState[index.stateIndex].visible);
          ImGui::TableNextColumn();
          if (metricDisplayState[index.stateIndex].visible) {
            if (ImGui::BeginCombo("##Select style", possibleAxis[metricDisplayState[index.stateIndex].axis], ImGuiComboFlags_NoArrowButton)) {
              for (int n = 0; n < IM_ARRAYSIZE(possibleAxis); n++) {
                bool is_selected = (metricDisplayState[index.stateIndex].axis == n);
                if (ImGui::Selectable(possibleAxis[n], is_selected)) {
                  metricDisplayState[index.stateIndex].axis = n;
                }
                if (is_selected) {
                  ImGui::SetItemDefaultFocus(); // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                }
              }
              ImGui::EndCombo();
            }
          }
          ImGui::TableNextColumn();
          ImGui::Text("%s/%s", node.label.c_str(), label.label);
          ImGui::PopID();
        }
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
  ImGui::SetNextWindowPos(ImVec2(metricDisplayPos, ImGui::GetIO().DisplaySize.y - state.bottomPaneSize), 0);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x - metricDisplayPos, state.bottomPaneSize), 0);

  ImGui::Begin("Devices", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

  if (!metricSelectorVisible) {
    metricSelectorVisible = ImGui::Button(">> Show metric selector");
  } else {
    metricSelectorVisible = !ImGui::Button("<< Hide metric selector");
  }
  static char const* plotStyles[] = {
    "lines",
    "histograms",
    "sparks",
    "table",
    "stems"};
  ImGui::SameLine();
  static enum MetricsDisplayStyle currentStyle = MetricsDisplayStyle::Lines;
  static char const* currentStyleStr = plotStyles[0];
  ImGui::TextUnformatted("Metric display style:");
  ImGui::SameLine();
  ImGui::PushItemWidth(100);
  if (ImGui::BeginCombo("##Select style", currentStyleStr)) {
    for (int n = 0; n < IM_ARRAYSIZE(plotStyles); n++) {
      bool is_selected = (currentStyleStr == plotStyles[n]); // You can store your selection however you want, outside or inside your objects
      if (ImGui::Selectable(plotStyles[n], is_selected)) {
        currentStyleStr = plotStyles[n];
        currentStyle = (MetricsDisplayStyle)n;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus(); // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
      }
    }
    ImGui::EndCombo();
  }
  ImGui::PopItemWidth();
  bool locked = false;

  size_t gmi = 0;
  int visibleMetrics = 0;
  static std::vector<int> visibleDevicesIndex;
  static std::vector<MetricIndex> visibleMetricsIndex;

  visibleDevicesIndex.reserve(totalMetrics);
  visibleDevicesIndex.clear();
  visibleMetricsIndex.clear();

  for (size_t si = 0; si < TOTAL_TYPES_OF_METRICS; ++si) {
    auto& metricsInfos = *metricsStore.metrics[si];
    for (size_t di = 0; di < metricsInfos.size(); ++di) {
      auto& metricInfo = metricsInfos[di];
      bool deviceVisible = false;
      for (size_t mi = 0; mi < metricInfo.metrics.size(); ++mi) {
        auto& label = metricInfo.metricLabels[mi];
        auto& state = metricDisplayState[gmi];
        if (state.visible) {
          deviceVisible = true;
          visibleMetrics++;
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
          maxTime = std::max(maxTime, curMaxTime);
          visibleMetricsIndex.push_back(MetricIndex{si, di, mi, gmi});
        }
        gmi++;
      }
      if (deviceVisible) {
        visibleDevicesIndex.push_back(di);
      }
    }
  }
  if (visibleMetricsIndex.empty()) {
    TextCenter("Please enable some metric.");
    ImGui::End();
    return;
  };

  switch (currentStyle) {
    case MetricsDisplayStyle::Stems:
    case MetricsDisplayStyle::Histos:
    case MetricsDisplayStyle::Lines: {
      displayDeviceMetrics("Metrics",
                           minTime, maxTime, 1024,
                           currentStyle, metricDisplayState, metricsStore);
    } break;
    case MetricsDisplayStyle::Sparks: {
      displaySparks(state.startTime, visibleMetricsIndex, metricDisplayState, metricsStore);
    } break;
    case MetricsDisplayStyle::Table: {
      static std::vector<float> visibleDevicesOffsets;
      visibleDevicesOffsets.clear();
      visibleDevicesOffsets.resize(visibleDevicesIndex.size());

      size_t lastDevice = -1;
      int visibleDeviceCount = -1;
      /// Calculate the size of all the metrics for a given device
      for (auto index : visibleMetricsIndex) {
        auto& metricsInfos = *metricsStore.metrics[index.storeIndex];
        if (lastDevice != index.deviceIndex) {
          visibleDeviceCount++;
          lastDevice = index.deviceIndex;
        }
        auto label = metricsInfos[index.deviceIndex].metricLabels[index.metricIndex].label;
        visibleDevicesOffsets[visibleDeviceCount] += ImGui::CalcTextSize(label, nullptr, true).x;
      }
      // The Device name header.
      if (ImGui::BeginTable("##metrics-table", visibleMetricsIndex.size() + 1, ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX, ImVec2{-1, -1})) {
        ImGui::TableSetupColumn("##close button", ImGuiTableColumnFlags_NoResize | ImGuiTableColumnFlags_WidthFixed, 20);
        for (auto index : visibleMetricsIndex) {
          ImGui::TableSetupColumn("##device-header", ImGuiTableColumnFlags_WidthFixed, 100);
        }
        ImGui::TableSetupScrollFreeze(1, 2);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("");
        visibleDeviceCount = -1;
        lastDevice = -1;
        for (auto index : visibleMetricsIndex) {
          ImGui::TableNextColumn();
          auto& metricsInfos = *metricsStore.metrics[index.storeIndex];
          auto& devices = *metricsStore.specs[index.storeIndex];
          auto& metric = metricsInfos[index.deviceIndex];
          auto label = metricsInfos[index.deviceIndex].metricLabels[index.metricIndex].label;
          if (lastDevice == index.deviceIndex) {
            continue;
          }
          visibleDeviceCount++;
          lastDevice = index.deviceIndex;
          auto& spec = devices[index.deviceIndex];

          ImGui::TextUnformatted(spec.label.c_str());
        }

        // The metrics headers
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("#");
        lastDevice = -1;
        for (auto index : visibleMetricsIndex) {
          ImGui::TableNextColumn();
          auto& metricsInfos = *metricsStore.metrics[index.storeIndex];
          auto& metric = metricsInfos[index.deviceIndex];
          auto label = metricsInfos[index.deviceIndex].metricLabels[index.metricIndex].label;
          ImGui::Text("%s (%" PRIu64 ")", label, (uint64_t)metric.metrics[index.metricIndex].filledMetrics);
        }

        // Calculate which columns we want to see.
        ImGuiListClipper clipper;
        // For now
        clipper.Begin(1024);

        while (clipper.Step()) {
          for (size_t i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            ImGui::TableNextRow();
            metricsTableRow(visibleMetricsIndex, metricsStore, i);
          }
        }
        ImGui::EndTable();
      }
    } break;
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

/// Display information window about the driver
/// and its state.
void displayDriverInfo(DriverInfo const& driverInfo, DriverControl& driverControl)
{
  ImGui::Begin("Driver information");
  static int pid = getpid();

  if (driverControl.state == DriverControlState::STEP) {
    driverControl.state = DriverControlState::PAUSE;
  }
  auto state = reinterpret_cast<int*>(&driverControl.state);
  ImGui::RadioButton("Play", state, static_cast<int>(DriverControlState::PLAY));
  ImGui::SameLine();
  ImGui::RadioButton("Pause", state, static_cast<int>(DriverControlState::PAUSE));
  ImGui::SameLine();
  ImGui::RadioButton("Step", state, static_cast<int>(DriverControlState::STEP));

  auto& registry = driverInfo.configContext->options();
  ImGui::Columns();

  ImGui::Text("PID: %d - Control port %d", pid, driverInfo.port);
  ImGui::Text("Frame cost (latency): %.1f(%.1f)ms", driverInfo.frameCost, driverInfo.frameLatency);
  ImGui::Text("Input parsing cost (latency): %.1f(%.1f)ms", driverInfo.inputProcessingCost, driverInfo.inputProcessingLatency);
  ImGui::Text("State stack (depth %lu)", driverInfo.states.size());
  if (ImGui::Button("SIGCONT all children")) {
    kill(0, SIGCONT);
  }
  ImGui::SameLine();
  if (ImGui::Button("Debug driver")) {
    std::string pidStr = std::to_string(pid);
    setenv("O2DEBUGGEDPID", pidStr.c_str(), 1);
#ifdef __APPLE__
    std::string defaultAppleDebugCommand =
      "osascript -e 'tell application \"Terminal\"'"
      " -e 'activate'"
      " -e 'do script \"lldb -p \" & (system attribute \"O2DEBUGGEDPID\") & \"; exit\"'"
      " -e 'end tell'";
    setenv("O2DPLDEBUG", defaultAppleDebugCommand.c_str(), 0);
#else
    setenv("O2DPLDEBUG", "xterm -hold -e gdb attach $O2DEBUGGEDPID &", 0);
#endif
    int retVal = system(getenv("O2DPLDEBUG"));
    (void)retVal;
  }

  ImGui::SameLine();
  if (ImGui::Button("Profile")) {
    std::string pidStr = std::to_string(pid);
    setenv("O2PROFILEDPID", pidStr.c_str(), 1);
#ifdef __APPLE__
    auto defaultAppleProfileCommand = fmt::format(
      "osascript -e 'tell application \"Terminal\"'"
      " -e 'activate'"
      " -e 'do script \"xcrun xctrace record --output dpl-profile-{0}.trace"
      " --time-limit 30s --template Time\\\\ Profiler --attach {0} "
      " && open dpl-profile-{0}.trace && exit\"'"
      " -e 'end tell'",
      pid);
    std::cout << defaultAppleProfileCommand << std::endl;
    setenv("O2DPLPROFILE", defaultAppleProfileCommand.c_str(), 0);
#else
    setenv("O2DPLPROFILE", "xterm -hold -e perf record -a -g -p $O2PROFILEDPID > perf-$O2PROFILEDPID.data &", 0);
#endif
    int retVal = system(getenv("O2DPLPROFILE"));
    (void)retVal;
  }

  for (size_t i = 0; i < driverInfo.states.size(); ++i) {
    ImGui::Text("#%lu: %s", i, DriverInfoHelper::stateToString(driverInfo.states[i]));
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
  guiState.bottomPaneSize = 340;
  guiState.leftPaneSize = 200;
  guiState.rightPaneSize = 300;

  // Show all the panes by default.
  guiState.bottomPaneVisible = true;
  guiState.leftPaneVisible = true;
  guiState.rightPaneVisible = true;
  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  guiState.startTime = now.tv_sec - ImGui::GetTime();
  std::vector<TopologyNodeInfo> deviceNodesInfos;
  for (auto& device : devices) {
    deviceNodesInfos.push_back(TopologyNodeInfo{device.id});
  }
  std::vector<TopologyNodeInfo> driverNodesInfos;
  driverNodesInfos.push_back(TopologyNodeInfo{"driver"});

  return [&guiState, &infos, &devices, &metadata, &controls, &metricsInfos, &driverInfo, &driverControl, deviceNodesInfos, driverNodesInfos]() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 0.;
    style.WindowRounding = 0.;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0x1b / 255.f, 0x1b / 255.f, 0x1b / 255.f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0x1b / 255.f, 0x1b / 255.f, 0x1b / 255.f, 1.00f);

    showTopologyNodeGraph(guiState, infos, devices, metadata, controls, metricsInfos);

    AllMetricsStore metricsStore;
    std::vector<DeviceMetricsInfo> driverMetrics{driverInfo.metrics};
    metricsStore.metrics[DEVICE_METRICS] = &metricsInfos;
    metricsStore.metrics[DRIVER_METRICS] = &driverMetrics;
    metricsStore.specs[DEVICE_METRICS] = &deviceNodesInfos;
    metricsStore.specs[DRIVER_METRICS] = &driverNodesInfos;
    displayMetrics(guiState, driverInfo, infos, metadata, controls, metricsStore);
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
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 3 * 2, i * windowPosStepping), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x / 3, ImGui::GetIO().DisplaySize.y - 300),
                                 ImGuiCond_Once);
        ImGui::Begin(state.label.c_str(), &control.logVisible);

        ImGui::InputText("Log filter", control.logFilter, sizeof(control.logFilter));
        ImGui::InputText("Log start trigger", control.logStartTrigger, sizeof(control.logStartTrigger));
        ImGui::InputText("Log stop trigger", control.logStopTrigger, sizeof(control.logStopTrigger));
        ImGui::Checkbox("Stop logging", &control.quiet);
        ImGui::SameLine();
        ImGui::Combo("Log level", reinterpret_cast<int*>(&control.logLevel), LogParsingHelpers::LOG_LEVELS,
                     (int)LogParsingHelpers::LogLevel::Size, 5);

        ImGui::Separator();
        ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetTextLineHeightWithSpacing()), false,
                          ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
        displayHistory(info, control);
        ImGui::EndChild();
        ImGui::End();
      }
      popWindowColorDueToStatus();
    }
  };
}

} // namespace o2::framework::gui
