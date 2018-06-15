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
#include "DriverControl.cxx"
#include "DriverInfo.cxx"
#include "Framework/FrameworkGUIDevicesGraph.h"
#include "Framework/PaletteHelpers.h"

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2
{
namespace framework
{

struct DeviceGUIState {
  std::string label;
};

struct WorkspaceGUIState {
  int selectedMetric;
  std::vector<std::string> availableMetrics;
  std::vector<DeviceGUIState> devices;
};

static WorkspaceGUIState gState;

void optionsTable(const DeviceSpec& spec, const DeviceControl& control)
{
  if (spec.options.empty()) {
    return;
  }
  if (ImGui::CollapsingHeader("Options:")) {
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
            ImGui::Text("\"%s\" (default)", option.defaultValue.get<const char*>());
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

// This is to display the information in the data relayer
struct HeatMapHelper {
  template <typename RECORD, typename ITEM>
  static void draw(const char* name,
                   float widgetYSize,
                   std::function<size_t()> const& getNumRecords,
                   std::function<RECORD(size_t)> const& getRecord,
                   std::function<size_t(RECORD const&)> const& getNumItems,
                   std::function<ITEM const&(RECORD const&, size_t)> const& getItem,
                   std::function<int(ITEM const&)> const& getValue,
                   std::function<ImU32(int value)> const& getColor)
  {
    ImU32 BORDER_COLOR = ImColor(200, 200, 200, 40);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 winPos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_sz = ImGui::GetWindowSize();
    ImGui::BeginChild(name, ImVec2(canvas_sz.x, widgetYSize), true);
    drawList->AddQuad(
      ImVec2(0., 0.) + winPos,
      ImVec2{ canvas_sz.x - 1, 0 } + winPos,
      ImVec2{ canvas_sz.x - 1, widgetYSize } + winPos,
      ImVec2{ 0, widgetYSize } + winPos,
      BORDER_COLOR);
    float padding = 1;
    for (size_t ri = 0, re = getNumRecords(); ri < re; ri++) {
      auto record = getRecord(ri);
      ImVec2 xOffset{ (ri * canvas_sz.x / getNumRecords()) + padding, 0 };
      ImVec2 xSize{ canvas_sz.x / getNumRecords() - 2 * padding, 0 };
      for (size_t mi = 0, me = getNumItems(record); mi < me; mi++) {
        ImVec2 yOffSet{ 0, (mi * widgetYSize / getNumItems(record)) + padding };
        ImVec2 ySize{ 0, (widgetYSize / getNumItems(record)) - 2 * padding };
        drawList->AddQuadFilled(
          xOffset + yOffSet + winPos,
          xOffset + xSize + yOffSet + winPos,
          xOffset + xSize + yOffSet + ySize + winPos,
          xOffset + yOffSet + ySize + winPos,
          getColor(getValue(getItem(record, mi))));
      }
    }
    ImGui::EndChild();
  }
};

bool startsWith(std::string mainStr, std::string toMatch)
{
  if (mainStr.find(toMatch) == 0) {
    return true;
  } else {
    return false;
  }
}

void displayDataRelayer(DeviceMetricsInfo const& metrics,
                        DeviceInfo const& info)
{
  auto& viewIndex = info.dataRelayerViewIndex;

  auto getNumRecords = [&viewIndex]() -> size_t {
    if (viewIndex.isComplete()) {
      return viewIndex.w;
    }
    return 0;
  };
  auto getRecord = [&metrics](size_t i) -> int {
    return i;
  };
  auto getNumItems = [&viewIndex](int record) {
    if (viewIndex.isComplete()) {
      return viewIndex.h;
    }
    return 0;
  };
  auto getItem = [&metrics, &viewIndex](int const& record, size_t i) {
    // Calculate the index in the viewIndex.
    auto idx = record * viewIndex.h + i;
    assert(viewIndex.indexes.size() > idx);
    MetricInfo const& metricInfo = metrics.metrics[viewIndex.indexes[idx]];
    assert(metrics.intMetrics.size() > metricInfo.storeIdx);
    auto& data = metrics.intMetrics[metricInfo.storeIdx];
    return data[(metricInfo.pos - 1) % data.size()];
  };
  auto getValue = [](int const& item) { return item; };
  auto getColor = [](int value) {
    const ImU32 SLOT_EMPTY = ImColor(20, 20, 20, 255);
    const ImU32 SLOT_FULL = ImColor(0xf9, 0xcd, 0xad, 255);
    const ImU32 SLOT_DISPATCHED = ImColor(0xc8, 0xc8, 0xa9, 255);
    const ImU32 SLOT_DONE = ImColor(0x83, 0xaf, 0, 0x9b);
    const ImU32 SLOT_ERROR = ImColor(0xfe, 0x43, 0x65, 255);
    switch (value) {
      case 0:
        return SLOT_EMPTY;
      case 1:
        return SLOT_FULL;
      case 2:
        return SLOT_DISPATCHED;
      case 3:
        return SLOT_DONE;
    }
    return SLOT_ERROR;
  };

  HeatMapHelper::draw<int, int>("DataRelayer",
                                100.f,
                                getNumRecords,
                                getRecord,
                                getNumItems,
                                getItem,
                                getValue,
                                getColor);
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
    if (strstr(line.c_str(), control.logFilter) != 0) {
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

void historyBar(DeviceGUIState& state, const DeviceSpec& spec, const DeviceMetricsInfo& metricsInfo)
{
  bool open = ImGui::TreeNode(state.label.c_str());
  if (open) {
    ImGui::Text("# channels: %lu", spec.outputChannels.size() + spec.inputChannels.size());
    ImGui::TreePop();
  }
  ImGui::NextColumn();

  if (gState.selectedMetric == -1) {
    ImGui::NextColumn();
    return;
  }

  auto currentMetricName = gState.availableMetrics[gState.selectedMetric];

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
      ImGui::PlotLines(currentMetricName.c_str(), getter, &data, data.size);
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
      ImGui::PlotLines(currentMetricName.c_str(), getter, &data, data.size);
      ImGui::NextColumn();
    } break;
    default:
      ImGui::NextColumn();
      return;
      break;
  }
}

void displayDeviceHistograms(const std::vector<DeviceInfo>& infos, const std::vector<DeviceSpec>& devices,
                             std::vector<DeviceControl>& controls, const std::vector<DeviceMetricsInfo>& metricsInfos)
{
  bool graphNodes = true;
  showTopologyNodeGraph(&graphNodes, infos, devices);
  ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - 300), 0);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 300), 0);

  // Calculate the unique set of metrics, as available in the metrics service
  std::set<std::string> allMetricsNames;
  for (const auto& metricsInfo : metricsInfos) {
    for (const auto& labelsPairs : metricsInfo.metricLabelsIdx) {
      allMetricsNames.insert(labelsPairs.first);
    }
  }
  using NamesIndex = std::vector<std::string>;
  gState.availableMetrics.clear();
  std::copy(allMetricsNames.begin(), allMetricsNames.end(), std::back_inserter(gState.availableMetrics));

  ImGui::Begin("Devices");
  ImGui::Combo("Select metric", &gState.selectedMetric,
               [](void* data, int idx, const char** outText) -> bool {
                 NamesIndex* v = reinterpret_cast<NamesIndex*>(data);
                 if (idx >= v->size()) {
                   return false;
                 }
                 *outText = v->at(idx).c_str();
                 return true;
               },
               &gState.availableMetrics, gState.availableMetrics.size());

  // Calculate the full timestamp range for the selected metric
  if (gState.selectedMetric >= 0) {
    auto currentMetricName = gState.availableMetrics[gState.selectedMetric];
    size_t minTime = -1;
    size_t maxTime = 0;
    for (auto& metricInfo : metricsInfos) {
      size_t mi = DeviceMetricsHelper::metricIdxByName(currentMetricName, metricInfo);
      if (mi == metricInfo.metricLabelsIdx.size()) {
        continue;
      }
      auto& metric = metricInfo.metrics[mi];
      size_t minRangePos = metric.pos % metricInfo.timestamps.size();
      size_t maxRangePos = (size_t)(metric.pos) - 1 % metricInfo.timestamps.size();
      auto& timestamps = metricInfo.timestamps[mi];
      size_t curMinTime = timestamps[minRangePos];
      size_t curMaxTime = timestamps[maxRangePos];
      minTime = minTime < curMinTime ? minTime : curMinTime;
      maxTime = maxTime > curMaxTime ? maxTime : curMaxTime;
    }
    ImGui::Text("min timestamp: %zu, max timestamp: %zu", minTime, maxTime);
  }
  ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetItemsLineHeightWithSpacing()), false,
                    ImGuiWindowFlags_HorizontalScrollbar);
  ImGui::Columns(2);
  ImGui::SetColumnOffset(1, 300);
  for (size_t i = 0; i < gState.devices.size(); ++i) {
    DeviceGUIState& guiState = gState.devices[i];
    const DeviceSpec& spec = devices[i];
    const DeviceMetricsInfo& metricsInfo = metricsInfos[i];

    historyBar(guiState, spec, metricsInfo);
  }
  ImGui::Columns(1);
  ImGui::EndChild();
  ImGui::End();
}

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

void pushWindowColorDueToStatus(const DeviceInfo& info)
{
  using LogLevel = LogParsingHelpers::LogLevel;

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

void popWindowColorDueToStatus() { ImGui::PopStyleColor(3); }

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

  auto &registry = driverInfo.configContext->options();
  ImGui::TextUnformatted("Workflow options:");
  ImGui::Columns(2);
  for (auto &option : driverInfo.workflowOptions) {
    ImGui::TextUnformatted(option.name.c_str());
    ImGui::NextColumn();
    switch (option.type) {
      case ConfigParamSpec::ParamType::Int64:
      case ConfigParamSpec::ParamType::Int:
        ImGui::Text("%d", registry.get<int>(option.name.c_str()));
        break;
      case ConfigParamSpec::ParamType::Float:
        ImGui::Text("%f", registry.get<float>(option.name.c_str()));
        break;
      case ConfigParamSpec::ParamType::Double:
        ImGui::Text("%f", registry.get<double>(option.name.c_str()));
        break;
      case ConfigParamSpec::ParamType::String:
        ImGui::Text("%s", registry.get<std::string>(option.name.c_str()).c_str());
        break;
      case ConfigParamSpec::ParamType::Bool:
        ImGui::TextUnformatted(registry.get<bool>(option.name.c_str()) ? "true" : "false");
        break;
      case ConfigParamSpec::ParamType::Empty:
      case ConfigParamSpec::ParamType::Unknown:
        break;
    }
    ImGui::NextColumn();
  }
  ImGui::Columns();

  ImGui::Text("State stack (depth %lu)", driverInfo.states.size());

  for (size_t i = 0; i < driverInfo.states.size(); ++i) {
    ImGui::Text("#%lu: %s", i, DriverHelper::stateToString(driverInfo.states[i]));
  }

  ImGui::End();
}

// FIXME: return empty function in case we were not built
// with GLFW support.
///
std::function<void(void)> getGUIDebugger(const std::vector<DeviceInfo>& infos, const std::vector<DeviceSpec>& devices,
                                         const std::vector<DeviceMetricsInfo>& metricsInfos,
                                         const DriverInfo& driverInfo, std::vector<DeviceControl>& controls,
                                         DriverControl& driverControl)
{
  gState.selectedMetric = -1;
  // FIXME: this should probaly have a better mapping between our window state and
  gState.devices.resize(infos.size());
  for (size_t i = 0; i < gState.devices.size(); ++i) {
    DeviceGUIState& state = gState.devices[i];
    state.label = devices[i].id + "(" + std::to_string(infos[i].pid) + ")";
  }

  return [&infos, &devices, &controls, &metricsInfos, &driverInfo, &driverControl]() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.09f, 0.09f, 1.00f);

    displayDeviceHistograms(infos, devices, controls, metricsInfos);
    displayDriverInfo(driverInfo, driverControl);

    int windowPosStepping = (ImGui::GetIO().DisplaySize.y - 500) / gState.devices.size();

    for (size_t i = 0; i < gState.devices.size(); ++i) {
      DeviceGUIState& state = gState.devices[i];
      assert(i < infos.size());
      assert(i < devices.size());
      const DeviceInfo& info = infos[i];
      const DeviceSpec& spec = devices[i];
      const DeviceMetricsInfo& metrics = metricsInfos[i];

      DeviceControl& control = controls[i];
      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 3 * 2, i * windowPosStepping), ImGuiSetCond_Once);
      ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x / 3, ImGui::GetIO().DisplaySize.y - 300),
                               ImGuiSetCond_Once);

      pushWindowColorDueToStatus(info);

      ImGui::Begin(state.label.c_str());
      if (ImGui::CollapsingHeader("Channels")) {
        ImGui::Text("# channels: %lu", spec.inputChannels.size() + spec.outputChannels.size());
        ChannelsTableHelper::channelsTable("Inputs:", spec.inputChannels);
        ChannelsTableHelper::channelsTable("Outputs:", spec.outputChannels);
      }
      optionsTable(spec, control);
      if (ImGui::CollapsingHeader("Data relayer")) {
        ImGui::Text("Completion policy: %s", spec.completionPolicy.name.c_str());
        displayDataRelayer(metrics, info);
      }
      if (ImGui::CollapsingHeader("Logs", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Stop logging", &control.quiet);
        ImGui::InputText("Log filter", control.logFilter, sizeof(control.logFilter));
        ImGui::InputText("Log start trigger", control.logStartTrigger, sizeof(control.logStartTrigger));
        ImGui::InputText("Log stop trigger", control.logStopTrigger, sizeof(control.logStopTrigger));
        ImGui::Combo("Log level", reinterpret_cast<int*>(&control.logLevel), LogParsingHelpers::LOG_LEVELS,
                     (int)LogParsingHelpers::LogLevel::Size, 5);

        ImGui::Separator();
        ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetItemsLineHeightWithSpacing()), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        displayHistory(info, control);
        ImGui::EndChild();
      }
      ImGui::End();
      popWindowColorDueToStatus();
    }
  };
}

} // namespace framework
} // namespace o2
