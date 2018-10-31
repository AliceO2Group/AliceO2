#ifndef IMGUI_EXTRAS_H
#define IMGUI_EXTRAS_H

#include "imgui.h"

namespace ImGui
{
// Start PlotMultiLines(...) and PlotMultiHistograms(...)------------------------
// by @JaapSuter and @maxint (please see: https://github.com/ocornut/imgui/issues/632)
static inline ImU32 InvertColorU32(ImU32 in)
{
  ImVec4 in4 = ColorConvertU32ToFloat4(in);
  in4.x = 1.f - in4.x;
  in4.y = 1.f - in4.y;
  in4.z = 1.f - in4.z;
  return GetColorU32(in4);
}

void PlotMultiLines(
  const char* label,
  int num_datas,
  const char** names,
  const ImColor* colors,
  float (*getter)(const void* data, int idx),
  const void* const* datas,
  int values_count,
  float scale_min,
  float scale_max,
  ImVec2 graph_size);

void PlotMultiHistograms(
  const char* label,
  int num_hists,
  const char** names,
  const ImColor* colors,
  float (*getter)(const void* data, int idx),
  const void* const* datas,
  int values_count,
  float scale_min,
  float scale_max,
  ImVec2 graph_size);

// End PlotMultiLines(...) and PlotMultiHistograms(...)--------------------------
} // namespace ImGui
#endif // IMGUI_EXTRAS_H
