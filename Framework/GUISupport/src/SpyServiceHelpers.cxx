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

#include <uv.h>
#include "SpyServiceHelpers.h"
#include "DebugGUI/imgui.h"
#include "Framework/GuiCallbackContext.h"
#include "SpyService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Headers/DataHeader.h"
#include <cinttypes>
#include <bitset>

namespace o2::framework
{
struct GUIData {
  const o2::header::DataHeader* header;
  std::vector<char*> dataParts;
  std::vector<size_t> dataLength;
};

void SpyServiceHelpers::webGUI(ServiceRegistryRef registry)
{
  auto& spy = registry.get<SpyService>();
  auto partsAlive = spy.partsAlive;
  if (partsAlive) {
    auto parts = spy.parts;

    static std::vector<GUIData> guiData;
    guiData.clear();

    int i = 0;

    while (i < parts->Size()) {
      std::string headerString((char*)(*parts)[i].GetData(), (*parts)[i].GetSize());

      GUIData guiDataPacket;

      auto header = o2::header::get<o2::header::DataHeader*>(headerString.c_str());
      guiDataPacket.header = header;

      int payloadParts = (int)header->splitPayloadParts;

      int lastPart = i + payloadParts;

      while (i < lastPart) {
        i++;
        guiDataPacket.dataParts.push_back((char*)(*parts)[i].GetData());
        guiDataPacket.dataLength.push_back((*parts)[i].GetSize());
      }

      guiData.push_back(guiDataPacket);

      i++;
    }

    int selectedFrame = spy.selectedFrame;
    int selectedData = spy.selectedData;
    const o2::header::DataHeader* header;

    header = guiData.size() ? guiData[selectedFrame].header : nullptr;

    if (header != nullptr) {
      ImGui::Begin(fmt::format("Spy {}", header->dataDescription.str).c_str());
      ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen);
      if (ImGui::Button("Skip 10s")) {
        auto loop = registry.get<DeviceState>().loop;
        spy.enableAfter = uv_now(loop) + 10000;
        uv_stop(loop);
      }

      static int selectedHeaderIndex = 0;

      if (ImGui::BeginCombo("Header", std::string(std::string(header->dataOrigin.str) + " " + std::string(header->dataDescription.str) + " " + std::to_string(selectedHeaderIndex)).c_str())) {
        int i = 0;
        for (const auto d : guiData) {
          // auto h = o2::header::get<o2::header::DataHeader*>(d.header);
          auto h = d.header;
          if (h) {
            std::string value = std::string(h->dataOrigin.str) + " " + std::string(h->dataDescription.str) + " " + std::to_string(i);
            // bool isSelected = selectedFrame == i;
            bool isSelected = selectedHeaderIndex == i;
            if (ImGui::Selectable(value.c_str(), isSelected)) {
              spy.selectedFrame = i;
              selectedHeaderIndex = i;
            }
            if (isSelected) {
              ImGui::SetItemDefaultFocus();
            }
          }
          i++;
        }
        ImGui::EndCombo();
      }
      ImGui::Text("Selected Frame Index: %d", selectedFrame);
      if (ImGui::CollapsingHeader("Header", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Columns(2);
        ImGui::Text("sMagicString: %d", header->sMagicString);
        ImGui::Text("sVersion: %d", header->sVersion);
        ImGui::Text("sHeaderType: %s", header->sHeaderType.str);
        ImGui::Text("sSerializationMethod: %s", header->sSerializationMethod.str);
        ImGui::NextColumn();
        ImGui::Text("Header size: %d", header->headerSize);
        ImGui::Text("Payload Size: %" PRIu64, header->payloadSize);
        ImGui::Text("Header Version: %d", header->headerVersion);
        ImGui::Text("flagsNextHeader: %d", header->flagsNextHeader);
        ImGui::Text("serialization: %s", header->serialization.str);
        ImGui::Text("Data Description: %s", header->dataDescription.str);
        ImGui::Text("Data Origin: %s", header->dataOrigin.str);
        ImGui::Text("Run Number: %d", header->runNumber);
        ImGui::Text("Sub Specification: %d", header->subSpecification);

        ImGui::Columns(1);
      }

      auto data = guiData[selectedFrame].dataParts[selectedData];
      auto size = guiData[selectedFrame].dataLength[selectedData];

      static int rowStart = 0;
      static int numRows = 10;

      static bool displayHex = false;

      if (ImGui::CollapsingHeader("Payload", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Rows", &numRows, 10, 40);
        ImGui::SliderInt("Starting Row", &rowStart, 0, size / 4);
        ImGui::Checkbox("Hexadecimal", &displayHex);
        ImGui::Text("Number of Rows, %lu", size / 4);
        ImGui::BeginChild("##ScrollingRegion", ImVec2(0, 430), false, ImGuiWindowFlags_HorizontalScrollbar);

        if (data != nullptr && size > 0 && ImGui::BeginTable("Data", 5, ImGuiTableFlags_Borders)) {
          ImGui::TableSetupColumn("");
          ImGui::TableSetupColumn("#0");
          ImGui::TableSetupColumn("#1");
          ImGui::TableSetupColumn("#2");
          ImGui::TableSetupColumn("#3");
          ImGui::TableHeadersRow();
          for (int row = rowStart; row < rowStart + numRows && row < size / 4; row++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%04d", row * 4);

            for (int column = 1; column < 5; column++) {
              std::stringstream hex;
              if (size > row * 4) {
                ImGui::TableSetColumnIndex(column);
                unsigned char dataElem = data[row * 4 + (column - 1)];
                if (displayHex) {
                  hex << std::hex << (int)dataElem;
                } else {
                  hex << std::bitset<8>{static_cast<unsigned long long>(dataElem)};
                }
              }
              std::string hexStr = hex.str();
              if (displayHex && hexStr.length() == 1) {
                hexStr.insert(0, "0");
              }
              ImGui::Text("%s", hexStr.c_str());
            }
          }
          ImGui::EndTable();
        }
        ImGui::EndChild();
      }
      ImGui::End();
    }
  }
}

} // namespace o2::framework
