// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FrameworkGUIDevicesGraph.h"
#include "Framework/FrameworkGUIDataRelayerUsage.h"
#include "Framework/FrameworkGUIState.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/PaletteHelpers.h"
#include "FrameworkGUIDeviceInspector.h"
#include "DebugGUI/imgui.h"
#include <algorithm>
#include <cmath>
#include <vector>

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x+rhs.x, lhs.y+rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x-rhs.x, lhs.y-rhs.y); }

namespace o2
{
namespace framework
{
namespace gui
{

struct NodeColor {
  ImVec4 normal;
  ImVec4 hovered;
  ImVec4 title;
  ImVec4 title_hovered;
};

using LogLevel = LogParsingHelpers::LogLevel;

NodeColor
decideColorForNode(const DeviceInfo &info) {
  NodeColor result;
  if (info.active == false) {
    result.normal = PaletteHelpers::RED;
    result.hovered = PaletteHelpers::SHADED_RED;
    return result;
  }
  switch(info.maxLogLevel) {
    case LogParsingHelpers::LogLevel::Error:
      result.normal = PaletteHelpers::SHADED_RED;
      result.hovered = PaletteHelpers::RED;
      result.title = PaletteHelpers::RED;
      result.title_hovered = PaletteHelpers::DARK_RED;
      break;
    case LogLevel::Warning:
      result.normal = PaletteHelpers::SHADED_YELLOW;
      result.hovered = PaletteHelpers::YELLOW;
      result.title = PaletteHelpers::YELLOW;
      result.title_hovered = PaletteHelpers::DARK_YELLOW;
      break;
    case LogLevel::Info:
      result.normal = PaletteHelpers::SHADED_GREEN;
      result.hovered = PaletteHelpers::GREEN;
      result.title = PaletteHelpers::GREEN;
      result.title_hovered = PaletteHelpers::DARK_GREEN;
      break;
    default:
      result.normal = PaletteHelpers::GRAY;
      result.hovered = PaletteHelpers::LIGHT_GRAY;
      result.title = PaletteHelpers::GRAY;
      result.title_hovered = PaletteHelpers::BLACK;
      break;
  }
  return result;
}

void showTopologyNodeGraph(WorkspaceGUIState &state,
                           const std::vector<DeviceInfo> &infos,
                           const std::vector<DeviceSpec> &specs,
                           std::vector<DeviceControl> &controls,
                           const std::vector<DeviceMetricsInfo>& metricsInfos
                          )
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), 0);
    if (state.bottomPaneVisible) {
      ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y - state.bottomPaneSize), 0);
    } else {
      ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y), 0);
    }

    ImGui::Begin("Physical topology view", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    // Dummy
    struct Node
    {
        int     ID;
        char    Name[32];
        ImVec2  Pos, Size;
        float   Value;
        ImVec4  Color;
        int     InputsCount, OutputsCount;

        Node(int id, const char* name, const ImVec2& pos, float value, const ImVec4& color, int inputs_count, int outputs_count) { ID = id; strncpy(Name, name, 31); Name[31] = 0; Pos = pos; Value = value; Color = color; InputsCount = inputs_count; OutputsCount = outputs_count; }

        ImVec2 GetInputSlotPos(int slot_no) const   { return ImVec2(Pos.x, Pos.y + Size.y * ((float)slot_no+1) / ((float)InputsCount+1)); }
        ImVec2 GetOutputSlotPos(int slot_no) const  { return ImVec2(Pos.x + Size.x, Pos.y + Size.y * ((float)slot_no+1) / ((float)OutputsCount+1)); }
    };
    struct NodeLink
    {
        int     InputIdx, InputSlot, OutputIdx, OutputSlot;

        NodeLink(int input_idx, int input_slot, int output_idx, int output_slot) { InputIdx = input_idx; InputSlot = input_slot; OutputIdx = output_idx; OutputSlot = output_slot; }
    };

    static ImVector<Node> nodes;
    static ImVector<NodeLink> links;
    static bool inited = false;
    static ImVec2 scrolling = ImVec2(0.0f, 0.0f);
    static bool show_grid = true;
    static int node_selected = -1;

    auto prepareChannelView = [&specs](ImVector<Node> &nodeList) {
      std::map<std::string, std::pair<int, int>> linkToIndex;
      for (int si = 0; si < specs.size(); ++si) {
        int oi = 0;
        for (auto &&output : specs[si].outputChannels) {
          linkToIndex.insert(std::make_pair(output.name, std::make_pair(si, oi)));
          oi += 1;
        }
      }
      for (int si = 0; si < specs.size(); ++si) {
        auto &spec = specs[si];
        // FIXME: display nodes using topological sort
        nodeList.push_back(Node(si, spec.id.c_str(),  ImVec2(40+120*si,50 + (120 * si) % 500), 0.5f,
                        ImColor(255,100,100),
                        spec.inputChannels.size(),
                        spec.outputChannels.size()));
        int ii = 0;
        for (auto &input : spec.inputChannels) {
          auto const &outName = input.name;
          auto const &out = linkToIndex.find(input.name);
          if (out == linkToIndex.end()) {
            LOG(ERROR) << "Could not find suitable node for " << outName;
            continue;
          }
          links.push_back(NodeLink{ out->second.first, out->second.second, si, ii});
          ii += 1;
        }
      }
    };

    if (!inited)
    {
      prepareChannelView(nodes);
      inited = true;
    }

    // Create our child canvas
    ImGui::BeginGroup();
    ImGui::Checkbox("Show grid", &show_grid);
    ImGui::SameLine();
    if (ImGui::Button("Center")) {
      scrolling = ImVec2(0., 0.);
    }
    ImGui::SameLine();
    if (state.leftPaneVisible == false && ImGui::Button("Show tree")) {
       state.leftPaneVisible = true;
    }
    if (state.leftPaneVisible == true && ImGui::Button("Hide tree")) {
      state.leftPaneVisible = false;
    }
    ImGui::SameLine();
    if (state.bottomPaneVisible == false && ImGui::Button("Show metrics")) {
       state.bottomPaneVisible = true;
    }
    if (state.bottomPaneVisible == true && ImGui::Button("Hide metrics")) {
      state.bottomPaneVisible = false;
    }
    ImGui::SameLine();
    if (state.rightPaneVisible == false && ImGui::Button("Show inspector")) {
       state.rightPaneVisible = true;
    }
    if (state.rightPaneVisible == true && ImGui::Button("Hide inspector")) {
      state.rightPaneVisible = false;
    }
    ImGui::Separator();
    ImGui::EndGroup();
    auto toolbarSize = ImGui::GetItemRectSize();
    // Draw a list of nodes on the left side
    bool open_context_menu = false;
    int node_hovered_in_list = -1;
    int node_hovered_in_scene = -1;
    if (state.leftPaneVisible) {
      ImGui::BeginChild("node_list", ImVec2(state.leftPaneSize, 0));
      ImGui::Text("Devices");
      ImGui::Separator();
      for (int node_idx = 0; node_idx < nodes.Size; node_idx++)
      {
        Node* node = &nodes[node_idx];
        ImGui::PushID(node->ID);
        if (ImGui::Selectable(node->Name, node->ID == node_selected)) {
          if (ImGui::IsMouseDoubleClicked(0)) {
            controls[node_selected].logVisible = true;
          }
          node_selected = node->ID;
        }
        if (ImGui::IsItemHovered())
        {
          node_hovered_in_list = node->ID;
          open_context_menu |= ImGui::IsMouseClicked(1);
        }
        ImGui::PopID();
      }
      ImGui::EndChild();
      ImGui::SameLine();
    }

    ImGui::BeginGroup();

    const float NODE_SLOT_RADIUS = 4.0f;
    const ImVec2 NODE_WINDOW_PADDING(8.0f, 8.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1,1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, (ImU32) ImColor(60,60,70,200));
    ImVec2 graphSize = ImGui::GetWindowSize();
    if (state.leftPaneVisible) {
      graphSize.x -= state.leftPaneSize;
    }
    if (state.rightPaneVisible) {
      graphSize.x -= state.rightPaneSize;
    }
    graphSize.y -= toolbarSize.y + 20;
    ImGui::BeginChild("scrolling_region", graphSize, true, ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PushItemWidth(graphSize.x);

    ImVec2 offset = ImGui::GetCursorScreenPos() - scrolling;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    // Number of layers we need. 2 per node, plus 2 for 
    // the background stuff.
    draw_list->ChannelsSplit((nodes.Size + 2)*2);

    // Display grid
    if (show_grid)
    {
        ImU32 GRID_COLOR = ImColor(200,200,200,40);
        float GRID_SZ = 64.0f;
        ImVec2 win_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_sz = ImGui::GetWindowSize();
        for (float x = fmodf(offset.x,GRID_SZ); x < canvas_sz.x; x += GRID_SZ)
            draw_list->AddLine(ImVec2(x,0.0f)+win_pos, ImVec2(x,canvas_sz.y)+win_pos, GRID_COLOR);
        for (float y = fmodf(offset.y,GRID_SZ); y < canvas_sz.y; y += GRID_SZ)
            draw_list->AddLine(ImVec2(0.0f,y)+win_pos, ImVec2(canvas_sz.x,y)+win_pos, GRID_COLOR);
    }

    // Display links
    draw_list->ChannelsSetCurrent(0); // Background
    for (int link_idx = 0; link_idx < links.Size; link_idx++)
    {
        NodeLink* link = &links[link_idx];
        Node* node_inp = &nodes[link->InputIdx];
        Node* node_out = &nodes[link->OutputIdx];
        ImVec2 p1 = offset + node_inp->GetOutputSlotPos(link->InputSlot);
        ImVec2 p2 = ImVec2(-3*NODE_SLOT_RADIUS, 0) + offset + node_out->GetInputSlotPos(link->OutputSlot);
        draw_list->AddBezierCurve(p1, p1+ImVec2(+50,0), p2+ImVec2(-50,0), p2, ImColor(200,200,100), 3.0f);
    }

    // Display nodes
    for (int node_idx = 0; node_idx < nodes.Size; node_idx++)
    {
      auto backgroundLayer = (node_idx + 1) * 2;
      auto foregroundLayer = (node_idx + 1) * 2 + 1;
      // Selected node goes to front
      if (node_selected == node_idx) {
        backgroundLayer = (nodes.Size + 1) * 2;
        foregroundLayer = (nodes.Size + 1) * 2 + 1;
      }
      Node* node = &nodes[node_idx];
      const DeviceInfo &info = infos[node_idx];

      ImGui::PushID(node->ID);
      ImVec2 node_rect_min = offset + node->Pos;

      // Display node contents first
      draw_list->ChannelsSetCurrent(foregroundLayer);
      bool old_any_active = ImGui::IsAnyItemActive();
      ImGui::SetCursorScreenPos(node_rect_min + NODE_WINDOW_PADDING);
      ImGui::BeginGroup(); // Lock horizontal position
      ImGui::Text("%s", node->Name);
      gui::displayDataRelayer(metricsInfos[node->ID], infos[node->ID], ImVec2(140., 90.));
      ImGui::EndGroup();

      // Save the size of what we have emitted and whether any of the widgets are being used
      bool node_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());
      float attemptX = std::max(ImGui::GetItemRectSize().x, 150.f);
      float attemptY = std::min(ImGui::GetItemRectSize().y, 128.f);
      node->Size = ImVec2(attemptX, attemptY) + NODE_WINDOW_PADDING + NODE_WINDOW_PADDING;
      ImVec2 node_rect_max = node_rect_min + node->Size;
      ImVec2 node_rect_title = node_rect_min + ImVec2(node->Size.x, 24);

      // Display node box
      draw_list->ChannelsSetCurrent(backgroundLayer); // Background
      ImGui::SetCursorScreenPos(node_rect_min);
      ImGui::InvisibleButton("node", node->Size);
      if (ImGui::IsItemHovered())
      {
        node_hovered_in_scene = node->ID;
        open_context_menu |= ImGui::IsMouseClicked(1);
        if (ImGui::IsMouseDoubleClicked(0)) {
          controls[node->ID].logVisible = true;
        }
      }
      bool node_moving_active = ImGui::IsItemActive();
      if (node_widgets_active || node_moving_active)
          node_selected = node->ID;
      if (node_moving_active && ImGui::IsMouseDragging(0))
          node->Pos = node->Pos + ImGui::GetIO().MouseDelta;
      if (ImGui::IsWindowHovered() && !node_moving_active && ImGui::IsMouseDragging(0))
          scrolling = scrolling - ImVec2(ImGui::GetIO().MouseDelta.x / 4.f, ImGui::GetIO().MouseDelta.y / 4.f);

      auto nodeBg = decideColorForNode(info);

      auto hovered = (node_hovered_in_list == node->ID || node_hovered_in_scene == node->ID || (node_hovered_in_list == -1 && node_selected == node->ID));
      ImVec4 nodeBgColor = hovered ? nodeBg.hovered : nodeBg.normal;
      ImVec4 nodeTitleColor = hovered ? nodeBg.title_hovered : nodeBg.title;
      ImU32 node_bg_color = ImGui::ColorConvertFloat4ToU32(nodeBgColor);
      ImU32 node_title_color = ImGui::ColorConvertFloat4ToU32(nodeTitleColor);

      draw_list->AddRectFilled(node_rect_min + ImVec2(3.f, 3.f), node_rect_max + ImVec2(3.f, 3.f), ImColor(0,0,0,70), 4.0f);
      draw_list->AddRectFilled(node_rect_min, node_rect_max, node_bg_color, 4.0f);
      draw_list->AddRectFilled(node_rect_min, node_rect_title, node_title_color, 4.0f);
      draw_list->AddRect(node_rect_min, node_rect_max, ImColor(100,100,100), 4.0f);
      for (int slot_idx = 0; slot_idx < node->InputsCount; slot_idx++) {
        auto color = ImColor(200,200,100);
        ImVec2 p1(-3*NODE_SLOT_RADIUS,NODE_SLOT_RADIUS), p2(-3*NODE_SLOT_RADIUS,-NODE_SLOT_RADIUS), p3(0,0);
        auto pp1 = p1 + offset + node->GetInputSlotPos(slot_idx);
        auto pp2 = p2 + offset + node->GetInputSlotPos(slot_idx);
        auto pp3 = p3 + offset + node->GetInputSlotPos(slot_idx);
        draw_list->AddTriangleFilled(pp1, pp2, pp3, color);
        draw_list->AddCircleFilled(offset + node->GetInputSlotPos(slot_idx), NODE_SLOT_RADIUS, ImColor(150,150,150,150));
      }
      for (int slot_idx = 0; slot_idx < node->OutputsCount; slot_idx++)
          draw_list->AddCircleFilled(offset + node->GetOutputSlotPos(slot_idx), NODE_SLOT_RADIUS, ImColor(150,150,150,150));

      ImGui::PopID();
    }
    draw_list->ChannelsMerge();

    // Open context menu
    if (!ImGui::IsAnyItemHovered() && ImGui::IsMouseHoveringWindow() && ImGui::IsMouseClicked(1))
    {
        node_selected = node_hovered_in_list = node_hovered_in_scene = -1;
        open_context_menu = true;
    }
    if (open_context_menu)
    {
        ImGui::OpenPopup("context_menu");
        if (node_hovered_in_list != -1)
            node_selected = node_hovered_in_list;
        if (node_hovered_in_scene != -1)
            node_selected = node_hovered_in_scene;
    }

    // Scrolling
    //if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive() && ImGui::IsMouseDragging(2, 0.0f))
    //    scrolling = scrolling - ImGui::GetIO().MouseDelta;

    ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    ImGui::EndGroup();

    if (state.rightPaneVisible) {
      ImGui::SameLine();
      ImGui::BeginGroup();
      ImGui::BeginChild("inspector");
      ImGui::TextUnformatted("Device Inspector");
      ImGui::Separator();
      if (node_selected != -1) {
        auto &spec = specs[node_selected];
        auto &control = controls[node_selected];
        if (state.rightPaneVisible) {
          gui::displayDeviceInspector(spec, control);
        }
      }
      else {
        ImGui::TextWrapped("Select a node in the topology to display information about it");
      }
      ImGui::EndChild();
      ImGui::EndGroup();
    }
    ImGui::End();
}

} // namespace gui
} // namespace framework
} // namespace o2
