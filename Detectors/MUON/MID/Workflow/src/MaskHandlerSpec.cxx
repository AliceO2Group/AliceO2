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

/// \file   MID/Workflow/src/MaskHandlerSpec.cxx
/// \brief  Processor to handle the masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   13 August 2021

#include "MIDWorkflow/MaskHandlerSpec.h"

#include <array>
#include <vector>
#include <map>
#include <chrono>
#include <gsl/gsl>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/MaskMaker.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class MaskHandlerDeviceDPL
{
 public:
  void init(o2::framework::InitContext& ic)
  {

    mOutFilename = ic.options().get<std::string>("mid-mask-outfile");

    auto stop = [this]() {
      printSummary();
    };
    ic.services().get<of::CallbackService>().set<of::CallbackService::Id::Stop>(stop);
  }

  void printSummary()
  {
    std::string name = "calib";
    o2::mid::ColumnDataToLocalBoard colToBoard;
    o2::mid::ROBoardConfigHandler roBoardCfgHandler;

    for (auto& masks : mMasksHandlers) {
      auto maskVec = masks.getMasks();
      colToBoard.process(maskVec);
      auto roMasks = colToBoard.getData();
      if (maskVec.empty()) {
        LOG(info) << "No problematic digit found in " << name << " events";
      } else {
        LOG(info) << "Problematic digits found in " << name << " events. Corresponding masks:";
        for (auto& mask : maskVec) {
          LOG(info) << mask;
        }
        std::cout << "\nCorresponding boards masks:" << std::endl;
        for (auto& board : roMasks) {
          std::cout << board << std::endl;
        }
      }

      if (!mOutFilename.empty()) {
        auto idx = mOutFilename.find_last_of("/");
        auto insertIdx = (idx == std::string::npos) ? 0 : idx + 1;
        std::string outFilename = mOutFilename;
        outFilename.insert(insertIdx, "_");
        outFilename.insert(insertIdx, name.c_str());
        auto fullMasks = o2::mid::makeDefaultMasks();
        for (auto& mask : fullMasks) {
          masks.applyMask(mask);
        }
        colToBoard.process(fullMasks);
        roBoardCfgHandler.updateMasks(colToBoard.getData());
        roBoardCfgHandler.write(outFilename.c_str());
      }

      name = "FET";
    }
  }

  void
    run(o2::framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    std::array<gsl::span<const ColumnData>, 2> masks;

    masks[0] = pc.inputs().get<gsl::span<ColumnData>>("mid_noise_mask");
    masks[1] = pc.inputs().get<gsl::span<ColumnData>>("mid_dead_mask");

    auto tAlgoStart = std::chrono::high_resolution_clock::now();

    bool isChanged = false;
    for (int itype = 0; itype < 2; ++itype) {
      ChannelMasksHandler chMasks;
      for (auto& mask : masks[itype]) {
        chMasks.setFromChannelMask(mask);
      }
      if (!(chMasks == mMasksHandlers[itype])) {
        mMasksHandlers[itype] = chMasks;
        isChanged = true;
      }
    }

    if (isChanged) {
      printSummary();
    }

    mTimerMaskHandler += std::chrono::high_resolution_clock::now() - tAlgoStart;

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
  }

 private:
  std::array<ChannelMasksHandler, 2> mMasksHandlers{}; ///< Output masks
  std::chrono::duration<double> mTimer{0};             ///< full timer
  std::chrono::duration<double> mTimerMaskHandler{0};  ///< mask handler timer
  std::string mOutFilename{};                          ///< output filename
};

framework::DataProcessorSpec getMaskHandlerSpec()
{
  std::vector<of::InputSpec> inputSpecs{
    of::InputSpec{"mid_noise_mask", header::gDataOriginMID, "MASKS", 1},
    of::InputSpec{"mid_dead_mask", header::gDataOriginMID, "MASKS", 2}};

  return of::DataProcessorSpec{
    "MIDMaskHandler",
    {inputSpecs},
    {},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::MaskHandlerDeviceDPL>()},
    of::Options{
      {"mid-mask-outfile", of::VariantType::String, "", {"Output filename"}}}};
}
} // namespace mid
} // namespace o2