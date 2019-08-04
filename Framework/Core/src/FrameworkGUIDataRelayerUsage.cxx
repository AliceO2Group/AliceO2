// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DebugGUI/imgui.h"
#include <functional>
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceInfo.h"
#include <iostream>
#include <cmath>

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2
{
namespace framework
{
namespace gui
{

// This is to display the information in the data relayer
struct HeatMapHelper {
  template <typename RECORD, typename ITEM>
  static void draw(const char* name,
                   ImVec2 const& sizeHint,
                   std::function<size_t()> const& getNumRecords,
                   std::function<RECORD(size_t)> const& getRecord,
                   std::function<size_t(RECORD const&)> const& getNumItems,
                   std::function<ITEM const&(RECORD const&, size_t)> const& getItem,
                   std::function<int(ITEM const&)> const& getValue,
                   std::function<ImU32(int value)> const& getColor,
                   std::function<void(int row, int column)> const& describeCell)
  {
    ImVec2 size = ImVec2(sizeHint.x, std::min(sizeHint.y, 16.f * getNumItems(0) + 2));
    ImU32 BORDER_COLOR = ImColor(200, 200, 200, 255);
    ImU32 BACKGROUND_COLOR = ImColor(20, 20, 20, 255);
    constexpr float MAX_BOX_X_SIZE = 16.f;
    constexpr float MAX_BOX_Y_SIZE = 16.f;
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 winPos = ImGui::GetCursorScreenPos();
    auto records = getNumRecords();
    auto boxSizeX = std::min(size.x / records, MAX_BOX_X_SIZE);

    ImGui::InvisibleButton("sensible area", ImVec2(size.x, size.y));
    if (ImGui::IsItemHovered()) {
      auto pos = ImGui::GetMousePos() - winPos;
      auto slot = std::lround(std::trunc(pos.x / size.x * records));
      auto row = std::lround(std::trunc(pos.y / size.y));
      describeCell(row, slot);
    }

    drawList->AddRectFilled(
      ImVec2(0., 0.) + winPos,
      ImVec2{size.x, size.y} + winPos,
      BACKGROUND_COLOR);
    drawList->AddRect(
      ImVec2(0., 0.) + winPos,
      ImVec2{size.x - 1, size.y} + winPos,
      BORDER_COLOR);
    float padding = 1;
    for (size_t ri = 0, re = getNumRecords(); ri < re; ri++) {
      auto record = getRecord(ri);
      ImVec2 xOffset{(ri * boxSizeX) + padding, 0};
      ImVec2 xSize{boxSizeX - 2 * padding, 0};
      auto boxSizeY = std::min(size.y / getNumItems(record), MAX_BOX_Y_SIZE);
      for (size_t mi = 0, me = getNumItems(record); mi < me; mi++) {
        ImVec2 yOffSet{0, (mi * boxSizeY) + padding};
        ImVec2 ySize{0, boxSizeY - 2 * padding};

        drawList->AddRectFilled(
          xOffset + yOffSet + winPos,
          xOffset + xSize + yOffSet + ySize + winPos,
          getColor(getValue(getItem(record, mi))));
      }
    }

    ImGui::SetCursorScreenPos(winPos + size);
  }
};

void displayDataRelayer(DeviceMetricsInfo const& metrics,
                        DeviceInfo const& info,
                        ImVec2 const& size)
{
  auto& viewIndex = info.dataRelayerViewIndex;
  auto& variablesIndex = info.variablesViewIndex;
  auto& queriesIndex = info.queriesViewIndex;

  auto getNumRecords = [&viewIndex]() -> size_t {
    if (viewIndex.isComplete()) {
      return viewIndex.w;
    }
    return 0;
  };
  auto getRecord = [&metrics](size_t i) -> int {
    return i;
  };
  auto getNumItems = [&viewIndex](int record) -> int {
    if (viewIndex.isComplete()) {
      return viewIndex.h;
    }
    return 0;
  };
  auto getItem = [&metrics, &viewIndex](int const& record, size_t i) -> int const& {
    // Calculate the index in the viewIndex.
    auto idx = record * viewIndex.h + i;
    assert(viewIndex.indexes.size() > idx);
    MetricInfo const& metricInfo = metrics.metrics[viewIndex.indexes[idx]];
    assert(metrics.intMetrics.size() > metricInfo.storeIdx);
    auto& data = metrics.intMetrics[metricInfo.storeIdx];
    return data[(metricInfo.pos - 1) % data.size()];
  };
  auto getValue = [](int const& item) -> int { return item; };
  auto getColor = [](int value) {
    const ImU32 SLOT_EMPTY = ImColor(70, 70, 70, 255);
    const ImU32 SLOT_FULL = ImColor(0xf9, 0xcd, 0xad, 255);
    const ImU32 SLOT_DISPATCHED = ImColor(0xc8, 0xc8, 0xa9, 255);
    const ImU32 SLOT_DONE = ImColor(0x83, 0xaf, 0, 255);
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
  auto describeCell = [&metrics, &variablesIndex, &queriesIndex](int input, int slot) -> void {
    ImGui::BeginTooltip();
    for (size_t vi = 0; vi < variablesIndex.w; ++vi) {
      auto idx = (slot * variablesIndex.w) + vi;
      assert(idx < variablesIndex.indexes.size());
      MetricInfo const& metricInfo = metrics.metrics[variablesIndex.indexes[idx]];
      assert(metricInfo.storeIdx < metrics.stringMetrics.size());
      //assert(metricInfo.type == MetricType::String);
      auto& data = metrics.stringMetrics[metricInfo.storeIdx];
      ImGui::Text("$%zu: %s", vi, data[(metricInfo.pos - 1) % data.size()].data);
    }
    ImGui::EndTooltip();
  };

  if (getNumRecords()) {
    HeatMapHelper::draw<int, int>("DataRelayer",
                                  size,
                                  getNumRecords,
                                  getRecord,
                                  getNumItems,
                                  getItem,
                                  getValue,
                                  getColor,
                                  describeCell);
  }
}

} // namespace gui
} // namespace framework
} // namespace o2
