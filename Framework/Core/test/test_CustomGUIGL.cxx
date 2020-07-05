// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/InputSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"

#include <DebugGUI/DebugGUI.h>
#include <DebugGUI/Sokol3DUtils.h>
#include <DebugGUI/imgui.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {"source",
     Inputs{},
     {
       OutputSpec{{"test"}, "TST", "A"},
     },
     AlgorithmSpec{
       adaptStateless([](DataAllocator& outputs) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
         auto& out = outputs.make<int>(OutputRef{"test", 0});
       })}},
    {"dest",
     Inputs{
       {"test", "TST", "A"}},
     Outputs{},
     AlgorithmSpec{adaptStateful(
       [](CallbackService& callbacks) {
         void* window = initGUI("A test window");
         auto count = std::make_shared<int>(0);
         sokol::init3DContext(window);

         auto guiCallback = [count]() {
           ImGui::Begin("Some sub-window");
           ImGui::Text("Counter value: %i", *count);
           ImGui::End();
           sokol::render3D();
         };
         callbacks.set(CallbackService::Id::ClockTick,
                       [count, window, guiCallback]() {
                    (*count)++; window ? pollGUI(window, guiCallback) : false; });
         return adaptStateless([count](ControlService& control) {
           if (*count > 1000) {
             control.readyToQuit(QuitRequest::All);
           }
         });
       })}}};
}
