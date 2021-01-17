// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FrameworkGUIDevicesGraph.h"
#include "FrameworkGUIDataRelayerUsage.h"
#include "FrameworkGUIState.h"
#include "PaletteHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/Logger.h"
#include "FrameworkGUIDeviceInspector.h"
#include "Framework/Logger.h"
#include "../src/WorkflowHelpers.h"
#include "DebugGUI/imgui.h"
#include <DebugGUI/icons_font_awesome.h>
#include <algorithm>
#include <cmath>
#include <vector>

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace o2::framework::gui
{
struct NodeColor {
  ImVec4 normal;
  ImVec4 hovered;
  ImVec4 title;
  ImVec4 title_hovered;
};

using LogLevel = LogParsingHelpers::LogLevel;

NodeColor
  decideColorForNode(const DeviceInfo& info)
{
  NodeColor result;
  if (info.active == false) {
    result.normal = PaletteHelpers::RED;
    result.hovered = PaletteHelpers::SHADED_RED;
    return result;
  }
  switch (info.streamingState) {
    case StreamingState::EndOfStreaming:
      result.normal = PaletteHelpers::SHADED_YELLOW;
      result.hovered = PaletteHelpers::YELLOW;
      result.title = PaletteHelpers::YELLOW;
      result.title_hovered = PaletteHelpers::DARK_YELLOW;
      break;
    case StreamingState::Idle:
      result.normal = PaletteHelpers::SHADED_GREEN;
      result.hovered = PaletteHelpers::GREEN;
      result.title = PaletteHelpers::GREEN;
      result.title_hovered = PaletteHelpers::DARK_GREEN;
      break;
    case StreamingState::Streaming:
    default:
      result.normal = PaletteHelpers::GRAY;
      result.hovered = PaletteHelpers::LIGHT_GRAY;
      result.title = PaletteHelpers::GRAY;
      result.title_hovered = PaletteHelpers::BLACK;
      break;
  }
  return result;
}

/// Displays a grid
void displayGrid(bool show_grid, ImVec2 offset, ImDrawList* draw_list)
{
  if (show_grid == false) {
    return;
  }
  ImU32 GRID_COLOR = ImColor(200, 200, 200, 40);
  float GRID_SZ = 64.0f;
  ImVec2 win_pos = ImGui::GetCursorScreenPos();
  ImVec2 canvas_sz = ImGui::GetWindowSize();
  for (float x = fmodf(offset.x, GRID_SZ); x < canvas_sz.x; x += GRID_SZ) {
    draw_list->AddLine(ImVec2(x, 0.0f) + win_pos, ImVec2(x, canvas_sz.y) + win_pos, GRID_COLOR);
  }
  for (float y = fmodf(offset.y, GRID_SZ); y < canvas_sz.y; y += GRID_SZ) {
    draw_list->AddLine(ImVec2(0.0f, y) + win_pos, ImVec2(canvas_sz.x, y) + win_pos, GRID_COLOR);
  }
}

#define MAX_GROUP_NAME_SIZE 128

// Private helper struct to keep track of node groups
struct Group {
  int ID;
  char name[MAX_GROUP_NAME_SIZE];
  size_t metadataId;
  Group(int id, char const* n, size_t mid)
  {
    ID = id;
    strncpy(name, n, MAX_GROUP_NAME_SIZE);
    name[MAX_GROUP_NAME_SIZE - 1] = 0;
    metadataId = mid;
  }
};

// Private helper struct for the graph model
struct Node {
  int ID;
  int GroupID;
  char Name[64];
  ImVec2 Size;
  float Value;
  ImVec4 Color;
  int InputsCount, OutputsCount;

  Node(int id, int groupID, char const* name, float value, const ImVec4& color, int inputs_count, int outputs_count)
  {
    ID = id;
    GroupID = groupID;
    strncpy(Name, name, 63);
    Name[63] = 0;
    Value = value;
    Color = color;
    InputsCount = inputs_count;
    OutputsCount = outputs_count;
  }
};

// Private helper struct for the layout of the graph
struct NodePos {
  ImVec2 pos;
  static ImVec2 GetInputSlotPos(ImVector<Node> const& infos, ImVector<NodePos> const& positions, int nodeId, int slot_no)
  {
    ImVec2 const& pos = positions[nodeId].pos;
    ImVec2 const& size = infos[nodeId].Size;
    float inputsCount = infos[nodeId].InputsCount;
    return ImVec2(pos.x, pos.y + size.y * ((float)slot_no + 1) / (inputsCount + 1));
  }
  static ImVec2 GetOutputSlotPos(ImVector<Node> const& infos, ImVector<NodePos> const& positions, int nodeId, int slot_no)
  {
    ImVec2 const& pos = positions[nodeId].pos;
    ImVec2 const& size = infos[nodeId].Size;
    float outputsCount = infos[nodeId].OutputsCount;
    return ImVec2(pos.x + size.x, pos.y + size.y * ((float)slot_no + 1) / (outputsCount + 1));
  }
};

// Private helper struct for the edges in the graph
struct NodeLink {
  int InputIdx, InputSlot, OutputIdx, OutputSlot;

  NodeLink(int input_idx, int input_slot, int output_idx, int output_slot)
  {
    InputIdx = input_idx;
    InputSlot = input_slot;
    OutputIdx = output_idx;
    OutputSlot = output_slot;
  }
};

void showTopologyNodeGraph(WorkspaceGUIState& state,
                           std::vector<DeviceInfo> const& infos,
                           std::vector<DeviceSpec> const& specs,
                           std::vector<DataProcessorInfo> const& metadata,
                           std::vector<DeviceControl>& controls,
                           std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  ImGui::SetNextWindowPos(ImVec2(0, 0), 0);
  if (state.bottomPaneVisible) {
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y - state.bottomPaneSize), 0);
  } else {
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y), 0);
  }

  ImGui::Begin("Physical topology view", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

  static ImVector<Node> nodes;
  static ImVector<Group> groups;
  static ImVector<NodeLink> links;
  static ImVector<NodePos> positions;

  static bool inited = false;
  static ImVec2 scrolling = ImVec2(0.0f, 0.0f);
  static bool show_grid = true;
  static int node_selected = -1;

  auto prepareChannelView = [&specs, &metadata](ImVector<Node>& nodeList, ImVector<Group>& groupList) {
    struct LinkInfo {
      int specId;
      int outputId;
    };
    std::map<std::string, LinkInfo> linkToIndex;
    for (int si = 0; si < specs.size(); ++si) {
      int oi = 0;
      for (auto&& output : specs[si].outputChannels) {
        linkToIndex.insert(std::make_pair(output.name, LinkInfo{si, oi}));
        oi += 1;
      }
    }
    // Prepare the list of groups.
    std::string workflow = "Ungrouped";
    int groupId = 0;
    for (size_t mi = 0; mi < metadata.size(); ++mi) {
      auto const& metadatum = metadata[mi];
      if (metadatum.executable == workflow) {
        continue;
      }
      workflow = metadatum.executable;
      char* groupBasename = strrchr(workflow.data(), '/');
      char const* groupName = groupBasename ? groupBasename + 1 : workflow.data();
      bool hasDuplicate = false;
      for (size_t gi = 0; gi < groupList.Size; ++gi) {
        if (strncmp(groupName, groupList[gi].name, MAX_GROUP_NAME_SIZE - 1) == 0) {
          hasDuplicate = true;
          break;
        }
      }
      if (hasDuplicate == false) {
        groupList.push_back(Group(groupId++, groupName, mi));
      }
    }
    // Do matching between inputs and outputs
    for (int si = 0; si < specs.size(); ++si) {
      auto& spec = specs[si];
      int groupId = 0;

      auto metadatum = std::find_if(metadata.begin(), metadata.end(),
                                    [&name = spec.name](DataProcessorInfo const& info) { return info.name == name; });

      for (size_t gi = 0; gi < groupList.Size; ++gi) {
        if (metadatum == metadata.end()) {
          break;
        }
        const char* groupName = strrchr(metadatum->executable.data(), '/');
        if (strncmp(groupList[gi].name, groupName ? groupName + 1 : metadatum->executable.data(), 127) == 0) {
          groupId = gi;
          break;
        }
      }
      nodeList.push_back(Node(si, groupId, spec.id.c_str(), 0.5f,
                              ImColor(255, 100, 100),
                              spec.inputChannels.size(),
                              spec.outputChannels.size()));
      int ii = 0;
      for (auto& input : spec.inputChannels) {
        auto const& outName = input.name;
        auto const& out = linkToIndex.find(input.name);
        if (out == linkToIndex.end()) {
          LOG(ERROR) << "Could not find suitable node for " << outName;
          continue;
        }
        links.push_back(NodeLink{out->second.specId, out->second.outputId, si, ii});
        ii += 1;
      }
    }

    // ImVector does boudary checks, so I bypass the case there is no
    // edges.
    std::vector<TopoIndexInfo> sortedNodes = {{0, 0}};
    if (links.size()) {
      sortedNodes = WorkflowHelpers::topologicalSort(specs.size(), &(links[0].InputIdx), &(links[0].OutputIdx), sizeof(links[0]), links.size());
    }
    // This is to protect for the cases in which there is a loop in the
    // definition of the inputs and of the outputs due to the
    // way the forwarding creates hidden dependencies between processes.
    // This should not happen, but apparently it does.
    for (auto di = 0; di < specs.size(); ++di) {
      auto fn = std::find_if(sortedNodes.begin(), sortedNodes.end(), [di](TopoIndexInfo const& info) {
        return di == info.index;
      });
      if (fn == sortedNodes.end()) {
        sortedNodes.push_back({(int)di, 0});
      }
    }
    assert(specs.size() == sortedNodes.size());
    /// We resort them again, this time with the added layer information
    std::sort(sortedNodes.begin(), sortedNodes.end());

    std::vector<int> layerEntries(1024, 0);
    std::vector<int> layerMax(1024, 0);
    for (auto& node : sortedNodes) {
      layerMax[node.layer < 1023 ? node.layer : 1023] += 1;
    }

    // FIXME: display nodes using topological sort
    // Update positions
    for (int si = 0; si < specs.size(); ++si) {
      auto& node = sortedNodes[si];
      assert(node.index == si);
      int xpos = 40 + 240 * node.layer;
      int ypos = 300 + (600 / (layerMax[node.layer] + 1)) * (layerEntries[node.layer] - layerMax[node.layer] / 2);
      positions.push_back(NodePos{ImVec2(xpos, ypos)});
      layerEntries[node.layer] += 1;
    }
  };

  if (!inited) {
    prepareChannelView(nodes, groups);
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
    ImGui::Text("Workflows %d", groups.Size);
    ImGui::Separator();
    for (int groupId = 0; groupId < groups.Size; groupId++) {
      Group* group = &groups[groupId];
      if (ImGui::TreeNodeEx(group->name, ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int node_idx = 0; node_idx < nodes.Size; node_idx++) {
          Node* node = &nodes[node_idx];
          if (node->GroupID != groupId) {
            continue;
          }
          ImGui::Indent(15);
          ImGui::PushID(node->ID);
          if (ImGui::Selectable(node->Name, node->ID == node_selected)) {
            if (ImGui::IsMouseDoubleClicked(0)) {
              controls[node_selected].logVisible = true;
            }
            node_selected = node->ID;
          }
          if (ImGui::IsItemHovered()) {
            node_hovered_in_list = node->ID;
            open_context_menu |= ImGui::IsMouseClicked(1);
          }
          ImGui::PopID();
          ImGui::Unindent(15);
        }
        ImGui::TreePop();
      }
    }
    ImGui::EndChild();
    ImGui::SameLine();
  }

  ImGui::BeginGroup();

  const float NODE_SLOT_RADIUS = 4.0f;
  const ImVec2 NODE_WINDOW_PADDING(8.0f, 8.0f);

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
#if defined(ImGuiCol_ChildWindowBg)
  ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, (ImU32)ImColor(60, 60, 70, 200));
#else
  ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImU32)ImColor(60, 60, 70, 200));
#endif
  ImVec2 graphSize = ImGui::GetWindowSize();
  if (state.leftPaneVisible) {
    graphSize.x -= state.leftPaneSize;
  }
  if (state.rightPaneVisible) {
    graphSize.x -= state.rightPaneSize;
  }
  graphSize.y -= toolbarSize.y + 20;
  ImGui::BeginChild("scrolling_region", graphSize, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PushItemWidth(graphSize.x);

  ImVec2 offset = ImGui::GetCursorScreenPos() - scrolling;
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  // Number of layers we need. 2 per node, plus 2 for
  // the background stuff.
  draw_list->ChannelsSplit((nodes.Size + 2) * 2);

  // Display grid
  displayGrid(show_grid, offset, draw_list);

  // Display links
  draw_list->ChannelsSetCurrent(0); // Background
  for (int link_idx = 0; link_idx < links.Size; link_idx++) {
    NodeLink* link = &links[link_idx];
    ImVec2 p1 = offset + NodePos::GetOutputSlotPos(nodes, positions, link->InputIdx, link->InputSlot);
    ImVec2 p2 = ImVec2(-3 * NODE_SLOT_RADIUS, 0) + offset + NodePos::GetInputSlotPos(nodes, positions, link->OutputIdx, link->OutputSlot);
    draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, ImColor(200, 200, 100), 3.0f);
  }

  // Display nodes
  for (int node_idx = 0; node_idx < nodes.Size; node_idx++) {
    auto backgroundLayer = (node_idx + 1) * 2;
    auto foregroundLayer = (node_idx + 1) * 2 + 1;
    // Selected node goes to front
    if (node_selected == node_idx) {
      backgroundLayer = (nodes.Size + 1) * 2;
      foregroundLayer = (nodes.Size + 1) * 2 + 1;
    }
    Node* node = &nodes[node_idx];
    NodePos* pos = &positions[node_idx];
    const DeviceInfo& info = infos[node_idx];

    ImGui::PushID(node->ID);
    ImVec2 node_rect_min = offset + pos->pos;

    // Display node contents first
    draw_list->ChannelsSetCurrent(foregroundLayer);
    bool old_any_active = ImGui::IsAnyItemActive();
    ImGui::SetCursorScreenPos(node_rect_min + NODE_WINDOW_PADDING);
    ImGui::BeginGroup(); // Lock horizontal position
    ImGui::TextUnformatted(node->Name);
    switch (info.maxLogLevel) {
      case LogLevel::Error:
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", ICON_FA_EXCLAMATION_CIRCLE);
        break;
      case LogLevel::Warning:
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0, 1, 1, 1), "%s", ICON_FA_EXCLAMATION_TRIANGLE);
        break;
      default:
        break;
    }
    gui::displayDataRelayer(metricsInfos[node->ID], infos[node->ID], ImVec2(140., 90.));
    ImGui::EndGroup();

    // Save the size of what we have emitted and whether any of the widgets are being used
    bool node_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());
    float attemptX = std::max(ImGui::GetItemRectSize().x, 150.f);
    float attemptY = std::min(ImGui::GetItemRectSize().y, 128.f);
    node->Size = ImVec2(attemptX, attemptY) + NODE_WINDOW_PADDING + NODE_WINDOW_PADDING;
    ImVec2 node_rect_max = node_rect_min + node->Size;
    ImVec2 node_rect_title = node_rect_min + ImVec2(node->Size.x, 24);

    if (node_rect_min.x > 20 + 2 * NODE_WINDOW_PADDING.x + state.leftPaneSize + graphSize.x) {
      ImGui::PopID();
      continue;
    }
    if (node_rect_min.y > 20 + 2 * NODE_WINDOW_PADDING.y + toolbarSize.y + graphSize.y) {
      ImGui::PopID();
      continue;
    }

    // Display node box
    draw_list->ChannelsSetCurrent(backgroundLayer); // Background
    ImGui::SetCursorScreenPos(node_rect_min);
    ImGui::InvisibleButton("node", node->Size);
    if (ImGui::IsItemHovered()) {
      node_hovered_in_scene = node->ID;
      open_context_menu |= ImGui::IsMouseClicked(1);
      if (ImGui::IsMouseDoubleClicked(0)) {
        controls[node->ID].logVisible = true;
      }
    }
    bool node_moving_active = ImGui::IsItemActive();
    if (node_widgets_active || node_moving_active) {
      node_selected = node->ID;
    }
    if (node_moving_active && ImGui::IsMouseDragging(0)) {
      pos->pos = pos->pos + ImGui::GetIO().MouseDelta;
    }
    if (ImGui::IsWindowHovered() && !node_moving_active && ImGui::IsMouseDragging(0)) {
      scrolling = scrolling - ImVec2(ImGui::GetIO().MouseDelta.x / 4.f, ImGui::GetIO().MouseDelta.y / 4.f);
    }

    auto nodeBg = decideColorForNode(info);

    auto hovered = (node_hovered_in_list == node->ID || node_hovered_in_scene == node->ID || (node_hovered_in_list == -1 && node_selected == node->ID));
    ImVec4 nodeBgColor = hovered ? nodeBg.hovered : nodeBg.normal;
    ImVec4 nodeTitleColor = hovered ? nodeBg.title_hovered : nodeBg.title;
    ImU32 node_bg_color = ImGui::ColorConvertFloat4ToU32(nodeBgColor);
    ImU32 node_title_color = ImGui::ColorConvertFloat4ToU32(nodeTitleColor);

    draw_list->AddRectFilled(node_rect_min + ImVec2(3.f, 3.f), node_rect_max + ImVec2(3.f, 3.f), ImColor(0, 0, 0, 70), 4.0f);
    draw_list->AddRectFilled(node_rect_min, node_rect_max, node_bg_color, 4.0f);
    draw_list->AddRectFilled(node_rect_min, node_rect_title, node_title_color, 4.0f);
    draw_list->AddRect(node_rect_min, node_rect_max, ImColor(100, 100, 100), 4.0f);
    for (int slot_idx = 0; slot_idx < node->InputsCount; slot_idx++) {
      auto color = ImColor(200, 200, 100);
      ImVec2 p1(-3 * NODE_SLOT_RADIUS, NODE_SLOT_RADIUS), p2(-3 * NODE_SLOT_RADIUS, -NODE_SLOT_RADIUS), p3(0, 0);
      auto pp1 = p1 + offset + NodePos::GetInputSlotPos(nodes, positions, node_idx, slot_idx);
      auto pp2 = p2 + offset + NodePos::GetInputSlotPos(nodes, positions, node_idx, slot_idx);
      auto pp3 = p3 + offset + NodePos::GetInputSlotPos(nodes, positions, node_idx, slot_idx);
      draw_list->AddTriangleFilled(pp1, pp2, pp3, color);
      draw_list->AddCircleFilled(offset + NodePos::GetInputSlotPos(nodes, positions, node_idx, slot_idx), NODE_SLOT_RADIUS, ImColor(150, 150, 150, 150));
    }
    for (int slot_idx = 0; slot_idx < node->OutputsCount; slot_idx++) {
      draw_list->AddCircleFilled(offset + NodePos::GetOutputSlotPos(nodes, positions, node_idx, slot_idx), NODE_SLOT_RADIUS, ImColor(150, 150, 150, 150));
    }

    ImGui::PopID();
  }
  draw_list->ChannelsMerge();

  // Open context menu
  if (!ImGui::IsAnyItemHovered() && ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) && ImGui::IsMouseClicked(1)) {
    node_selected = node_hovered_in_list = node_hovered_in_scene = -1;
    open_context_menu = true;
  }
  if (open_context_menu) {
    ImGui::OpenPopup("context_menu");
    if (node_hovered_in_list != -1) {
      node_selected = node_hovered_in_list;
    }
    if (node_hovered_in_scene != -1) {
      node_selected = node_hovered_in_scene;
    }
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
      auto& node = nodes[node_selected];
      auto& spec = specs[node_selected];
      auto& control = controls[node_selected];
      auto& info = infos[node_selected];
      auto& metrics = metricsInfos[node_selected];
      auto& group = groups[node.GroupID];
      auto& metadatum = metadata[group.metadataId];

      if (state.rightPaneVisible) {
        gui::displayDeviceInspector(spec, info, metrics, metadatum, control);
      }
    } else {
      ImGui::TextWrapped("Select a node in the topology to display information about it");
    }
    ImGui::EndChild();
    ImGui::EndGroup();
  }
  ImGui::End();
}

} // namespace o2::framework::gui
