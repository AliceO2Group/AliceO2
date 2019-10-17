// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework GUITests
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "DebugGUI/imgui.h"
#include "Framework/FrameworkGUIDataRelayerUsage.h"
#include "Framework/FrameworkGUIDevicesGraph.h"
#include "Framework/FrameworkGUIState.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"
#include "../src/FrameworkGUIDeviceInspector.h"

BOOST_AUTO_TEST_CASE(SimpleGUITest)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();

  // Build atlas
  unsigned char* tex_pixels = nullptr;
  int tex_w, tex_h;
  io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);

  for (int n = 0; n < 50; n++) {
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();

    static float f = 0.0f;
    ImGui::Text("Hello, world!");
    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

    ImGui::Render();
  }

  ImGui::DestroyContext();
}

BOOST_AUTO_TEST_CASE(HeatmapTest)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();

  // Build atlas
  unsigned char* tex_pixels = nullptr;
  int tex_w, tex_h;
  io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);

  using namespace o2::framework;

  DeviceMetricsInfo metrics;
  DeviceInfo deviceInfo;
  DeviceInfo deviceInfo2;
  deviceInfo2.dataRelayerViewIndex = {
    "data_relayer",
    1,
    1,
    {0}};

  ImVec2 size(100, 100);

  for (int n = 0; n < 50; n++) {
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();

    gui::displayDataRelayer(metrics, deviceInfo, size);
    ParsedMetricMatch match;
    std::vector<std::string> dummyHeatmapMetrics = {
      "[METRIC] data_relayer/w,0 2 1789372894 hostname=test.cern.ch"
      "[METRIC] data_relayer/h,0 2 1789372894 hostname=test.cern.ch"
      "[METRIC] data_relayer/0,0 1 1789372894 hostname=test.cern.ch"
      "[METRIC] data_relayer/0,1 1 1789372894 hostname=test.cern.ch"
      "[METRIC] data_relayer/0,2 1 1789372894 hostname=test.cern.ch"
      "[METRIC] data_relayer/0,3 1 1789372894 hostname=test.cern.ch"};
    for (auto& metric : dummyHeatmapMetrics) {
      auto result = DeviceMetricsHelper::parseMetric(metric, match);
      // Add the first metric to the store
      result = DeviceMetricsHelper::processMetric(match, metrics);
    }
    gui::displayDataRelayer(metrics, deviceInfo2, size);

    ImGui::Render();
  }

  ImGui::DestroyContext();
}

BOOST_AUTO_TEST_CASE(DeviceInspector)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();

  // Build atlas
  unsigned char* tex_pixels = nullptr;
  int tex_w, tex_h;
  io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);

  using namespace o2::framework;
  DeviceSpec spec;
  DeviceInfo info;
  DeviceControl control;
  DeviceMetricsInfo metrics;
  DataProcessorInfo metadata = DataProcessorInfo{
    "foo",
    "bar",
    {},
    {}};

  for (int n = 0; n < 50; n++) {
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    gui::displayDeviceInspector(spec, info, metrics, metadata, control);
    ImGui::Render();
  }
  ImGui::DestroyContext();
}

BOOST_AUTO_TEST_CASE(DevicesGraph)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();

  // Build atlas
  unsigned char* tex_pixels = nullptr;
  int tex_w, tex_h;
  io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);

  using namespace o2::framework;
  gui::WorkspaceGUIState state;
  std::vector<DeviceSpec> specs;
  specs.push_back(
    DeviceSpec{
      "foo",
      "foo",
      {},
      {},
      {},
      {ConfigParamSpec{"global-config", VariantType::String, {"A global config option for all processor specs"}},
       ConfigParamSpec{"a-boolean", VariantType::Bool, true, {"A boolean which we pick by default"}},
       ConfigParamSpec{"an-int", VariantType::Int, 10, {"An int for which we pick up the default"}},
       ConfigParamSpec{"a-double", VariantType::Double, 11., {"A double for which we pick up the override"}}},
      AlgorithmSpec{},
      {},
      {},
      {},
      0,
      1,
      0,
      CompletionPolicy{}});

  std::vector<DeviceInfo> infos;
  infos.push_back(
    DeviceInfo{
      1000,
      0,
      1024,
      LogParsingHelpers::LogLevel::Error,
      std::vector<std::string>(1024, ""),
      std::vector<LogParsingHelpers::LogLevel>(1024, LogParsingHelpers::LogLevel::Info),
      "some error",
      {},
      true,
      false,
      StreamingState::Streaming,
      Metric2DViewIndex{}});

  std::vector<DataProcessorInfo> metadata;
  metadata.emplace_back(
    DataProcessorInfo{
      "foo",
      "bar",
      {},
      {}});

  std::vector<DeviceControl> controls;
  controls.push_back(
    DeviceControl{});
  std::vector<DeviceMetricsInfo> metrics;

  for (int n = 0; n < 50; n++) {
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    gui::showTopologyNodeGraph(state, infos, specs, metadata, controls, metrics);
    ImGui::Render();
  }
  ImGui::DestroyContext();
}
