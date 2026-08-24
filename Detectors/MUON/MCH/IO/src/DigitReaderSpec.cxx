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

/// \file   MCH/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MCH digits reader device
/// \author Michael Winn <Michael.Winn at cern.ch>
/// \date   17 April 2021

#include "MCHIO/DigitReaderSpec.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <gsl/span>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "CommonDataFormat/IRFrame.h"
#include "CommonUtils/IRFrameSelector.h"
#include "CommonUtils/StringUtils.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHDigitFiltering/DigitFilterParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace mch
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC) : mUseMC(useMC)
  {
    if (mUseMC) {
      mLabels = std::make_unique<TTreeReaderValue<dataformats::MCTruthContainer<MCCompLabel>>>(mTreeReader, "MCHMCLabels");
    }
  }

  void init(InitContext& ic)
  {
    auto fileName = utils::Str::concat_string(utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("mch-digit-infile"));
    connectTree(fileName);

    if (ic.options().hasOption("ignore-irframes") && !ic.options().get<bool>("ignore-irframes")) {
      mUseIRFrames = true;
    }

    mTimeOffset = ic.options().get<bool>("no-time-offset") ? 0 : DigitFilterParam::Instance().timeOffset;
  }

  void run(ProcessingContext& pc)
  {
    if (mUseIRFrames) {
      sendNextIRFrames(pc);
    } else {
      sendNextTF(pc);
    }
  }

 private:
  TTreeReader mTreeReader{};
  TTreeReaderValue<std::vector<ROFRecord>> mRofs = {mTreeReader, "MCHROFRecords"};
  TTreeReaderValue<std::vector<Digit>> mDigits = {mTreeReader, "MCHDigit"};
  std::unique_ptr<TTreeReaderValue<dataformats::MCTruthContainer<MCCompLabel>>> mLabels{};
  bool mUseMC = true;
  bool mUseIRFrames = false;
  int mTimeOffset = 0;

  void connectTree(std::string fileName)
  {
    auto file = TFile::Open(fileName.c_str());
    if (!file || file->IsZombie()) {
      throw std::invalid_argument(fmt::format("Opening file {} failed", fileName));
    }

    auto tree = file->Get<TTree>("o2sim");
    if (!tree) {
      throw std::invalid_argument(fmt::format("Tree o2sim not found in {}", fileName));
    }
    mTreeReader.SetTree(tree);
    mTreeReader.Restart();
  }

  void sendNextTF(ProcessingContext& pc)
  {
    // A timeframe holds no collision at all whenever the interaction rate is low enough, and the
    // digit tree then has no entry. Send empty containers and finish, rather than throwing.
    if (mTreeReader.GetEntries() == 0) {
      LOG(info) << "digit tree has no entry, sending empty output";
      pc.outputs().snapshot(OutputRef{"rofs"}, std::vector<ROFRecord>{});
      pc.outputs().snapshot(OutputRef{"digits"}, std::vector<Digit>{});
      if (mUseMC) {
        pc.outputs().snapshot(OutputRef{"labels"}, dataformats::MCTruthContainer<MCCompLabel>{});
      }
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    // load the next TF and check its validity (missing branch, ...)
    if (!mTreeReader.Next()) {
      throw std::invalid_argument(mTreeReader.fgEntryStatusText[mTreeReader.GetEntryStatus()]);
    }

    // send the whole TF
    pc.outputs().snapshot(OutputRef{"rofs"}, *mRofs);
    pc.outputs().snapshot(OutputRef{"digits"}, *mDigits);
    if (mUseMC) {
      pc.outputs().snapshot(OutputRef{"labels"}, **mLabels);
    }

    // stop here if it was the last one
    if (mTreeReader.GetCurrentEntry() + 1 >= mTreeReader.GetEntries()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }

  void sendNextIRFrames(ProcessingContext& pc)
  {
    std::vector<ROFRecord> rofs{};
    std::vector<Digit> digits{};
    dataformats::MCTruthContainer<MCCompLabel> labels{};

    // get the IR frames to select
    auto irFrames = pc.inputs().get<gsl::span<dataformats::IRFrame>>("driverInfo");

    if (!irFrames.empty()) {
      utils::IRFrameSelector irfSel{};
      irfSel.setSelectedIRFrames(irFrames, 0, 0, -mTimeOffset, true);
      const auto irMin = irfSel.getIRFrames().front().getMin();
      const auto irMax = irfSel.getIRFrames().back().getMax();

      // load the first TF if not already done
      bool loadNextTF = mTreeReader.GetCurrentEntry() < 0;

      while (true) {
        // load the next TF if requested
        if (loadNextTF && !mTreeReader.Next()) {
          throw std::invalid_argument(mTreeReader.fgEntryStatusText[mTreeReader.GetEntryStatus()]);
        }

        // look for selected ROFs in this TF and copy them
        if (!mRofs->empty() && mRofs->front().getBCData() <= irMax &&
            mRofs->back().getBCData() + mRofs->back().getBCWidth() - 1 >= irMin) {
          for (const auto& rof : *mRofs) {
            if (irfSel.check({rof.getBCData(), rof.getBCData() + rof.getBCWidth() - 1}) != -1) {
              rofs.emplace_back(rof);
              rofs.back().setDataRef(digits.size(), rof.getNEntries());
              digits.insert(digits.end(), mDigits->begin() + rof.getFirstIdx(), mDigits->begin() + rof.getFirstIdx() + rof.getNEntries());
              if (mUseMC) {
                for (auto i = 0; i < rof.getNEntries(); ++i) {
                  labels.addElements(labels.getIndexedSize(), (*mLabels)->getLabels(rof.getFirstIdx() + i));
                }
              }
            }
          }
        }

        // move to the next TF if needed and if any
        if ((mRofs->empty() || mRofs->back().getBCData() + mRofs->back().getBCWidth() - 1 < irMax) &&
            mTreeReader.GetCurrentEntry() + 1 < mTreeReader.GetEntries()) {
          loadNextTF = true;
          continue;
        }

        break;
      }
    }

    // send the selected data
    pc.outputs().snapshot(OutputRef{"rofs"}, rofs);
    pc.outputs().snapshot(OutputRef{"digits"}, digits);
    if (mUseMC) {
      pc.outputs().snapshot(OutputRef{"labels"}, labels);
    }

    // stop here if they were the last IR frames to select
    if (irFrames.empty() || irFrames.back().isLast()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
};

DataProcessorSpec getDigitReaderSpec(bool useMC, std::string_view specName,
                                     std::string_view outputDigitDataDescription,
                                     std::string_view outputDigitRofDataDescription,
                                     std::string_view outputDigitLabelDataDescription)
{
  std::string output = fmt::format("digits:MCH/{}/0;rofs:MCH/{}/0", outputDigitDataDescription, outputDigitRofDataDescription);
  if (useMC) {
    output += fmt::format(";labels:MCH/{}/0", outputDigitLabelDataDescription);
  }

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    std::string(specName),
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitsReaderDeviceDPL>(useMC)},
    Options{{"mch-digit-infile", VariantType::String, "mchdigits.root", {"Name of the input file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}},
            {"no-time-offset", VariantType::Bool, false, {"no time offset between IRFrames and digits"}}}};
}
} // namespace mch
} // namespace o2
