#ifndef IMGUI_EXTRAS_H
#define IMGUI_EXTRAS_H

#include "Framework/CompilerBuiltins.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }

namespace ImGui
{
// Start PlotMultiLines(...) and PlotMultiHistograms(...)------------------------
// by @JaapSuter and @maxint (please see: https://github.com/ocornut/imgui/issues/632)
// I assume the work is public domain.

// Look for the first entry larger in t than the current position
enum struct BinFunction {
  CEIL,
  FLOOR,
  MIN,
  MAX,
  AVG,
  COUNT
};

enum struct ValueScale {
  LINEAR,
  LOG
};

static inline ImU32 InvertColorU32(ImU32 in)
{
  ImVec4 in4 = ColorConvertU32ToFloat4(in);
  in4.x = 1.f - in4.x;
  in4.y = 1.f - in4.y;
  in4.z = 1.f - in4.z;
  return GetColorU32(in4);
}

static float binner(BinFunction binFunction,
                    const void* const* datas,
                    size_t data_idx,
                    size_t values_count,
                    float binStart, float binEnd,
                    size_t domain_min, size_t domain_max, float scale,
                    float (*getterY)(const void* data, int idx),
                    size_t (*getterX)(const void* data, int idx),
                    size_t& cacheXIdx,
                    size_t& count)
{
  float value;
  switch (binFunction) {
    case BinFunction::CEIL:
    case BinFunction::MAX:
      value = -FLT_MAX;
      break;
    case BinFunction::FLOOR:
    case BinFunction::MIN:
      value = FLT_MAX;
      break;
    default:
      value = 0;
  }
  // Accumulate the matching datapoints.
  while (cacheXIdx < values_count) {
    size_t bin = getterX(datas[data_idx], cacheXIdx);
    if ((bin < domain_min)) {
      cacheXIdx++;
      continue;
    }
    if ((bin > domain_max)) {
      return value;
    }
    // We are withing (domain_min, domain_max), what we have have below
    // works.
    float binPos = ImSaturate(float(bin - domain_min) * scale);
    if ((binPos < binStart)) {
      cacheXIdx++;
      continue;
    }
    if ((binPos > binEnd)) {
      return value;
    }
    count++;
    float binValue = getterY(datas[data_idx], cacheXIdx);
    cacheXIdx++;
    switch (binFunction) {
      case BinFunction::CEIL:
        value = std::max(value, ceilf(binValue));
        break;
      case BinFunction::MAX:
        value = std::max(value, binValue);
        break;
      case BinFunction::FLOOR:
        value = std::min(value, floorf(binValue));
        break;
      case BinFunction::MIN:
        value = std::min(value, binValue);
        break;
      case BinFunction::COUNT:
        value = count;
        break;
      case BinFunction::AVG:
        // NOTE: this will not actually work with too many values because of floating point
        value = (value * (count - 1) / count) + (binValue / count);
        break;
    }
  }
  return value;
}

void drawAxis(ImRect inner_bb, ImRect frame_bb,
              float scale_min, float scale_max,
              size_t domain_min, size_t domain_max,
              ImGuiWindow* window,
              const char* label,
              int res_w, float t_step)
{
  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;

  for (int n = 0; n < res_w; n += 10) {
    float binStart = n * t_step;
    ImVec2 binP0(binStart, 1);
    ImVec2 binP1(binStart, (n % 100 == 0) ? 0.95 : 0.98);
    ImVec2 binPos0 = ImLerp(inner_bb.Min, inner_bb.Max, binP0);
    ImVec2 binPos1 = ImLerp(inner_bb.Min, inner_bb.Max, binP1);
    window->DrawList->AddLine(binPos0, binPos1, ImColor(255, 255, 255, 200));
  }

  char buf[128];
  RenderText(ImVec2(frame_bb.Min.x + style.ItemInnerSpacing.x, inner_bb.Max.y), (snprintf(buf, 128, "%zu", domain_min), buf));
  RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 102, inner_bb.Max.y), (snprintf(buf, 128, "%zu", domain_max), buf));
  if (scale_min != scale_max) {
    RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 92, inner_bb.Min.y), (snprintf(buf, 128, "%.2f", scale_max), buf));
  }
  RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 92, inner_bb.Max.y - 12), (snprintf(buf, 128, "%.2f", scale_min), buf));
  window->DrawList->AddLine(ImVec2(frame_bb.Min.x + style.ItemInnerSpacing.x, inner_bb.Max.y),
                            ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 100, inner_bb.Max.y),
                            ImColor(255, 255, 255, 255));
  window->DrawList->AddLine(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 100, inner_bb.Min.y),
                            ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x - 100, inner_bb.Max.y),
                            ImColor(255, 255, 255, 255));
  RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);
};

struct FrameLayout {
  ImRect frame_bb;
  ImRect inner_bb;
  ImRect total_bb;
};

void addPlotWidget(ImGuiWindow* window, const char* label, FrameLayout& result, ImVec2& graph_size)
{

  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;

  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  if (graph_size.x == 0.0f)
    graph_size.x = CalcItemWidth();
  if (graph_size.y == 0.0f)
    graph_size.y = label_size.y + (style.FramePadding.y * 2);

  const ImVec2 axis_labels(100, 20);
  result.frame_bb = ImRect(window->DC.CursorPos, window->DC.CursorPos + ImVec2(graph_size.x, graph_size.y));
  result.inner_bb = ImRect(result.frame_bb.Min + style.FramePadding, result.frame_bb.Max - axis_labels - style.FramePadding);
  result.total_bb = ImRect(result.frame_bb.Min, result.frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
  ItemSize(result.total_bb, style.FramePadding.y);
  if (!ItemAdd(result.total_bb, 0))
    return;
  auto bg_color = GetColorU32(ImGuiCol_FrameBg);
  bg_color = ImColor{20, 20, 20, 255};
  RenderFrame(result.frame_bb.Min, result.frame_bb.Max, bg_color, true, style.FrameRounding);
};

static void PlotMultiEx(
  ImGuiPlotType plot_type,
  const char* label,
  int num_datas,
  const char** names,
  const ImColor* colors,
  float (*getterY)(const void* data, int idx),
  size_t (*getterX)(const void* data, int idx),
  const void* const* datas,
  int values_count,
  float scale_min,
  float scale_max,
  size_t domain_min,
  size_t domain_max,
  ImVec2 graph_size)
{
  const int values_offset = 0;

  FrameLayout layout;
  ImGuiWindow* window = GetCurrentWindow();
  ImGuiContext& g = *GImGui;
  if (window->SkipItems)
    return;
  addPlotWidget(window, label, layout, graph_size);

  // Use this to save the data below the current cursor position.
  std::vector<float> hoveredData(num_datas, 0);

  int domain_delta = domain_max - domain_min;
  float domain_scale = 1.f / (domain_max - domain_min);
  int res_w = ImMin((int)graph_size.x, domain_delta) + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);
  int item_count = values_count + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);

  // Tooltip on hover
  int v_hovered = -1;
  if (IsItemHovered()) {
    const float t = ImClamp((g.IO.MousePos.x - layout.inner_bb.Min.x) / (layout.inner_bb.Max.x - layout.inner_bb.Min.x), 0.0f, 1.f);
    int v_idx = (int)(t * item_count);
    // Make results numerically stable.
    if (v_idx >= values_count) {
      v_idx = values_count - 1;
    }
    IM_ASSERT(v_idx >= 0 && v_idx < values_count);

    v_hovered = v_idx;
    ImGui::BeginTooltip();
    ImGui::Text("Timestamp: %li", v_idx + domain_min);
    ImGui::EndTooltip();
  }

  const float t_step = 1.0f / (float)res_w;
  const float bin_size = domain_delta / (float)res_w;

  float lastValidValue = FLT_MAX;
  auto findFirstValidIndexAndValue = [&domain_min, &getterX, &values_count](void const* const data) {
    // Look for the first X value within the domain.
    int n;
    size_t vx0;
    for (n = 0; n < values_count; n++) {
      vx0 = getterX(data, n);
      if (vx0 >= domain_min) {
        return std::tie(n, vx0);
      }
    }
    return std::tie(values_count, vx0);
  };

  std::vector<ImVec2> path;
  path.reserve(res_w);
  // We iterate on all the data series.
  for (int data_idx = 0; data_idx < num_datas; ++data_idx) {
    auto [first_valid_index, vx0] = findFirstValidIndexAndValue(datas[data_idx]);
    size_t cacheXIdx = first_valid_index;
    float v0 = getterY(datas[data_idx], (first_valid_index + values_offset) % values_count);
    float t0 = ImSaturate((vx0 - domain_min) * domain_scale);
    ImVec2 tp0 = ImVec2(t0, 1.0f - ImSaturate((v0 - scale_min) / (scale_max - scale_min))); // Point in the normalized space of our target rectangle

    const ImU32 col_base = colors[data_idx];
    const ImU32 col_hovered = InvertColorU32(colors[data_idx]);

    ValueScale valueScale = ValueScale::LINEAR;

    auto scale = [valueScale](float v) {
      switch (valueScale) {
        case ValueScale::LINEAR:
          return v;
          break;
        case ValueScale::LOG:
          return logf(v);
      }
      O2_BUILTIN_UNREACHABLE();
    };

    float hoveredBinPos = (float)(v_hovered) / (float)(item_count);
    ImVec2 hoveredPos0;
    ImVec2 hoveredPos1;
    // We iterate on all the items.
    for (int n = 0; n < res_w; n++) {
      float binStart = n * t_step;
      float binEnd = (n + 1) * t_step;

      auto binFunction = BinFunction::MAX;

      size_t count = 0;
      float value = binner(binFunction,
                           datas,
                           data_idx,
                           values_count,
                           binStart, binEnd,
                           domain_min, domain_max, domain_scale,
                           getterY,
                           getterX,
                           cacheXIdx,
                           count);
      if (count == 0) {
        value = lastValidValue;
      } else {
        lastValidValue = value;
      }

      value = scale(value);

      const float t1 = n * t_step;
      const float v1 = value;
      // n is in resolution space, we transform it to v1_idx which is
      // in values_space.
      const size_t v1_idx = (float)n / (float)res_w * (float)values_count;

      if (hoveredBinPos > binStart && hoveredBinPos < binEnd) {
        hoveredData[data_idx] = value;
      }
      const ImVec2 tp1 = ImVec2(t1, 1.0f - ImSaturate((v1 - scale_min) / (scale_max - scale_min)));

      // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
      ImVec2 pos0 = ImLerp(layout.inner_bb.Min, layout.inner_bb.Max, tp0);
      ImVec2 pos1 = ImLerp(layout.inner_bb.Min, layout.inner_bb.Max, (plot_type == ImGuiPlotType_Lines) ? tp1 : ImVec2(tp1.x, 1.0f));

      if (plot_type == ImGuiPlotType_Lines) {
        if (path.empty() || memcmp(&path[path.size() - 1], &pos0, 8) != 0) {
          path.push_back(pos0);
        } else if (path[path.size() - 1].y == pos0.y) {
          path[path.size() - 1].x = pos0.x;
        } else if (path[path.size() - 1].x == pos0.x) {
          path[path.size() - 1].y = pos0.y;
        }
        if (path.empty() || memcmp(&path[path.size() - 1], &pos1, 8) != 0) {
          path.push_back(pos1);
        } else if (path[path.size() - 1].y == pos1.y) {
          path[path.size() - 1].x = pos1.x;
        } else if (path[path.size() - 1].x == pos1.x) {
          path[path.size() - 1].y = pos1.y;
        }
        if (v_hovered == v1_idx) {
          hoveredPos0 = pos0;
          hoveredPos1 = pos1;
        }
      } else if (plot_type == ImGuiPlotType_Histogram) {
        if (pos1.x >= pos0.x + 2.0f)
          pos1.x -= 1.0f;
        window->DrawList->AddRectFilled(pos0, pos1, v_hovered == v1_idx ? col_hovered : col_base);
      }

      t0 = t1;
      tp0 = tp1;
    }
    if (plot_type == ImGuiPlotType_Lines) {
      window->DrawList->AddPolyline(path.data(), path.size(), col_base, false, 1);
      path.clear();
      window->DrawList->AddLine(hoveredPos0, hoveredPos1, col_hovered, 3.f);
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    for (size_t data_idx = 0; data_idx < num_datas; data_idx++) {
      if (hoveredData[data_idx] == FLT_MAX) {
        continue;
      }
      if (plot_type == ImGuiPlotType_Lines) {
        TextColored(colors[data_idx], "%8.4f | %s", hoveredData[data_idx], names[data_idx]);
      } else if (plot_type == ImGuiPlotType_Histogram) {
        TextColored(colors[data_idx], "%8.4f | %s", hoveredData[data_idx], names[data_idx]);
      }
    }
    ImGui::EndTooltip();
  }

  drawAxis(layout.inner_bb, layout.frame_bb, scale_min, scale_max, domain_min, domain_max, window, label, res_w, t_step);
}

void PlotMultiLines(
  const char* label,
  int num_datas,
  const char** names,
  const ImColor* colors,
  float (*getterY)(const void* data, int idx),
  size_t (*getterX)(const void* data, int idx),
  const void* const* datas,
  int values_count,
  float scale_min,
  float scale_max,
  size_t domain_min,
  size_t domain_max,
  ImVec2 graph_size)
{
  PlotMultiEx(ImGuiPlotType_Lines, label, num_datas, names, colors, getterY, getterX, datas, values_count, scale_min, scale_max, domain_min, domain_max, graph_size);
}

void PlotMultiHistograms(
  const char* label,
  int num_hists,
  const char** names,
  const ImColor* colors,
  float (*getterY)(const void* data, int idx),
  size_t (*getterX)(const void* data, int idx),
  const void* const* datas,
  int values_count,
  float scale_min,
  float scale_max,
  size_t domain_min,
  size_t domain_max,
  ImVec2 graph_size)
{
  PlotMultiEx(ImGuiPlotType_Histogram, label, num_hists, names, colors, getterY, getterX, datas, values_count, scale_min, scale_max, domain_min, domain_max, graph_size);
}

// End PlotMultiLines(...) and PlotMultiHistograms(...)--------------------------
} // namespace ImGui
#endif // IMGUI_EXTRAS_H
