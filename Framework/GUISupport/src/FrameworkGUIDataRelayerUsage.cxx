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
#include "DebugGUI/imgui.h"
#include <functional>
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataProcessingStates.h"
#include "InspectorHelpers.h"
#include "PaletteHelpers.h"
#include "FrameworkGUIDataRelayerUsage.h"
#include <cstring>
#include <cmath>

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2::framework::gui
{
// This is to display the information in the data relayer
struct HeatMapHelper {
  template <typename RECORD, typename ITEM>
  static void draw(const char* /*name*/,
                   int& v,
                   ImVec2 const& sizeHint,
                   std::function<size_t()> const& getNumInputs,
                   std::function<size_t()> const& getNumRecords,
                   std::function<RECORD(size_t)> const& getRecord,
                   std::function<size_t(RECORD const&)> const& getNumItems,
                   std::function<ITEM const*(RECORD const&, size_t)> const& getItem,
                   std::function<int(ITEM const&)> const& getValue,
                   std::function<ImU32(int value)> const& getColor,
                   std::function<void(int row, int column)> const& describeCell)
  {
    float padding = 1;
    // add slider to scroll between the grid display windows
    size_t nw = getNumRecords() / WND;
    ImGui::PushItemWidth(sizeHint.x);
    ImGui::SliderInt("##window", &v, 1, nw, "wnd: %d", ImGuiSliderFlags_AlwaysClamp);
    ImVec2 sliderMin = ImGui::GetItemRectMin();

    constexpr float MAX_BOX_X_SIZE = 16.f;
    constexpr float MAX_BOX_Y_SIZE = 16.f;

    ImVec2 size = ImVec2(sizeHint.x, std::min(sizeHint.y, MAX_BOX_Y_SIZE * getNumItems(0) + 2));
    ImU32 BORDER_COLOR = ImColor(200, 200, 200, 255);
    ImU32 BACKGROUND_COLOR = ImColor(20, 20, 20, 255);
    ImU32 BORDER_COLOR_A = ImColor(200, 200, 200, 0);
    ImU32 BACKGROUND_COLOR_A = ImColor(0, 0, 0, 0);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 winPos = sliderMin;

    // overlay activity indicator on the slider
    auto xsz = size.x / nw;
    drawList->AddRectFilled(
      ImVec2{0., 0.} + winPos,
      ImVec2{size.x, size.y} + winPos,
      BACKGROUND_COLOR_A);
    drawList->AddRect(
      ImVec2{0. - 1, -1} + winPos,
      ImVec2{size.x + 1, size.y - 1} + winPos,
      BORDER_COLOR_A);

    const static auto colorA = ImColor(ImVec4{0.945, 0.096, 0.278, 0.5});
    const static auto colorE = ImColor(ImVec4{0, 0, 0, 0});

    drawList->PrimReserve(nw * 6, nw * 4);
    for (size_t iw = 0; iw < nw; ++iw) {
      ImVec2 xOffset{iw * xsz + 2 * padding, 0};
      ImVec2 xSize{xsz - 2 * padding, 0};
      ImVec2 yOffset{0, 2 * padding};
      ImVec2 ySize{0, 16 - 4 * padding};
      bool active = 0;
      for (size_t ir = iw; ir < ((iw + WND > getNumRecords()) ? getNumRecords() : iw + WND); ++ir) {
        for (size_t i = 0; i < getNumItems(ir); ++i) {
          active = getValue(*getItem(ir, i)) > 0;
          if (active) {
            break;
          }
        }
      }
      drawList->PrimRect(
        xOffset + yOffset + winPos,
        xOffset + xSize + yOffset + ySize + winPos,
        active ? colorA : colorE);
    }

    // display the grid
    size_t recordsWindow = v * WND;
    auto boxSizeX = std::min(size.x / WND, MAX_BOX_X_SIZE);
    auto numInputs = getNumInputs();
    winPos = ImGui::GetCursorScreenPos() + ImVec2{0, 7};
    ImGui::InvisibleButton("sensible area", ImVec2(size.x, size.y));
    if (ImGui::IsItemHovered()) {
      auto pos = ImGui::GetMousePos() - winPos;
      auto slot = (v - 1) * WND + std::lround(std::trunc(pos.x / size.x * WND));
      auto row = std::lround(std::trunc(pos.y / size.y * numInputs));
      describeCell(row, slot);
    }

    drawList->AddRectFilled(
      ImVec2(0., 0.) + winPos,
      ImVec2{size.x, size.y} + winPos,
      BACKGROUND_COLOR);
    drawList->AddRect(
      ImVec2(0. - 1, -1) + winPos,
      ImVec2{size.x + 1, size.y - 1} + winPos,
      BORDER_COLOR);

    size_t totalRects = 0;
    for (size_t ri = (v - 1) * WND; ri < recordsWindow; ri++) {
      auto record = getRecord(ri);
      totalRects += getNumItems(record);
    }

    drawList->PrimReserve(totalRects * 6, totalRects * 4);
    for (size_t ri = (v - 1) * WND; ri < recordsWindow; ri++) {
      auto record = getRecord(ri);
      ImVec2 xOffset{((ri - (v - 1) * WND) * boxSizeX) + padding, 0};
      ImVec2 xSize{boxSizeX - 2 * padding, 0};
      auto me = getNumItems(record);
      auto boxSizeY = std::min(size.y / me, MAX_BOX_Y_SIZE);
      for (size_t mi = 0; mi < me; mi++) {
        ImVec2 yOffSet{0, (mi * boxSizeY) + padding};
        ImVec2 ySize{0, boxSizeY - 2 * padding};

        drawList->PrimRect(
          xOffset + yOffSet + winPos,
          xOffset + xSize + yOffSet + ySize + winPos,
          getColor(getValue(*getItem(record, mi))));
      }
    }

    ImGui::SetCursorScreenPos(winPos + size);
  }
};

void displayDataRelayer(DeviceMetricsInfo const& /*metrics*/,
                        DeviceInfo const& /*info*/,
                        DeviceSpec const& spec,
                        DataProcessingStates const& states,
                        ImVec2 const& size,
                        int& v)
{
  auto getNumInputs = [&states]() -> size_t {
    auto& inputsView = states.statesViews[(int)ProcessingStateId::DATA_QUERIES];
    std::string_view inputs(states.statesBuffer.data() + inputsView.first, inputsView.size);
    if (inputs.size() == 0) {
      return 0;
    }
    // count the number of semi-colon separators to get number of inputs
    int numInputs = std::count(inputs.begin(), inputs.end(), ';');
    return numInputs;
  };
  auto getNumRecords = [&states]() -> size_t {
    auto& view = states.statesViews[(int)ProcessingStateId::DATA_RELAYER_BASE];
    if (view.size == 0) {
      return 0;
    }
    // The first number is the size of the pipeline
    int numRecords = strtoul(states.statesBuffer.data() + view.first, nullptr, 10);
    return numRecords;
  };
  auto getRecord = [](size_t i) -> int {
    return i;
  };
  auto getNumItems = [&states](int record) -> int {
    auto& view = states.statesViews[(int)ProcessingStateId::DATA_RELAYER_BASE + record];
    if (view.size == 0) {
      return 0;
    }
    char const* beginData = strchr(states.statesBuffer.data() + view.first, ' ') + 1;
    //  The number of elements is given by the size of the state, minus the header
    int size = view.size - (beginData - (states.statesBuffer.data() + view.first));
    return size;
  };
  auto getItem = [&states](int const& record, size_t i) -> int8_t const* {
    static int8_t const zero = '0';
    static int8_t const error = '4';
    char const *buffer = states.statesBuffer.data();
    auto& view = states.statesViews[(int)ProcessingStateId::DATA_RELAYER_BASE + record];
    if (view.size == 0) {
      return &zero;
    }
    char const* const beginData = strchr(buffer + view.first, ' ') + 1;
    // Protect against buffer overflows
    if ((size_t)view.size <= beginData - buffer + i - view.first) {
      return &error;
    }
    return (int8_t const*)beginData + i; };
  auto getValue = [](int8_t const& item) -> int { return item - '0'; };
  auto getColor = [](int value) {
    static const ImU32 SLOT_EMPTY = ImColor(70, 70, 70, 255);
    static const ImU32 SLOT_FULL = ImColor(PaletteHelpers::RED);
    static const ImU32 SLOT_DISPATCHED = ImColor(PaletteHelpers::YELLOW);
    static const ImU32 SLOT_DONE = ImColor(PaletteHelpers::GREEN);
    static const ImU32 SLOT_ERROR = ImColor(0xfe, 0x43, 0x65, 255);
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
  auto describeCell = [&states, &spec](int row, int slot) -> void {
    ImGui::BeginTooltip();

    // display the input (origin/descr/subspec)
    auto& inputsView = states.statesViews[(int)ProcessingStateId::DATA_QUERIES];
    std::string_view inputs(states.statesBuffer.data() + inputsView.first, inputsView.size);
    auto beginInputs = inputs.begin();
    auto endInputs = beginInputs + inputsView.size;
    char const* input = beginInputs;
    size_t i = 0;
    while (input != endInputs) {
      auto end = std::find(input, endInputs, ';');
      if ((end - input) == 0) {
        continue;
      }
      if (i == (size_t)row) {
        ImGui::Text("%d %.*s (%s)", row, int(end - input), input, InspectorHelpers::getLifeTimeStr(spec.inputs[i].matcher.lifetime));
        break;
      }
      ++i;
      input = end + 1;
    }

    // display context variables
    ImGui::Text("Input query matched values for slot: %d", slot);
    auto& view = states.statesViews[(short)ProcessingStateId::CONTEXT_VARIABLES_BASE + (short)slot];
    auto begin = view.first;
    for (size_t vi = 0; vi < data_matcher::MAX_MATCHING_VARIABLE; ++vi) {
      std::string_view state(states.statesBuffer.data() + begin, view.size);
      // find the semi-colon, which separates entries in the variable list
      auto pos = state.find(';');
      std::string_view value = state.substr(0, pos);
      // Do not display empty values.
      if (value.empty()) {
        begin += 1;
        continue;
      }
      switch (vi) {
        case o2::framework::data_matcher::STARTTIME_POS:
          ImGui::Text("$%zu (startTime): %.*s", vi, (int)value.size(), value.data());
          break;
        case o2::framework::data_matcher::TFCOUNTER_POS:
          ImGui::Text("$%zu (tfCounter): %.*s", vi, (int)value.size(), value.data());
          break;
        case o2::framework::data_matcher::FIRSTTFORBIT_POS:
          ImGui::Text("$%zu (firstTForbit): %.*s", vi, (int)value.size(), value.data());
          break;
        default:
          ImGui::Text("$%zu: %.*s", vi, (int)value.size(), value.data());
      }
      begin += pos + 1;
    }
    ImGui::EndTooltip();
  };

  if (getNumRecords()) {
    HeatMapHelper::draw<int, int8_t>("DataRelayer",
                                     v,
                                     size,
                                     getNumInputs,
                                     getNumRecords,
                                     getRecord,
                                     getNumItems,
                                     getItem,
                                     getValue,
                                     getColor,
                                     describeCell);
  }
}

} // namespace o2::framework::gui
