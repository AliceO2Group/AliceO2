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

/// \file   MID/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MID digits reader device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2019

#include "MIDWorkflow/DigitReaderSpec.h"

#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Variant.h"
#include "CommonDataFormat/IRFrame.h"
#include "CommonUtils/IRFrameSelector.h"
#include "CommonUtils/StringUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace mid
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC) : mUseMC(useMC)
  {
    if (mUseMC) {
      mLabels = std::make_unique<TTreeReaderValue<dataformats::MCTruthContainer<MCLabel>>>(mTreeReader, "MIDDigitMCLabels");
    }
  }

  void init(InitContext& ic)
  {
    auto filename = utils::Str::concat_string(utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("mid-digit-infile"));

    connectTree(filename);

    if (ic.options().hasOption("ignore-irframes") && !ic.options().get<bool>("ignore-irframes")) {
      mUseIRFrames = true;
    }
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
  TTreeReaderValue<std::vector<ROFRecord>> mRofs = {mTreeReader, "MIDROFRecords"};
  TTreeReaderValue<std::vector<ColumnData>> mDigits = {mTreeReader, "MIDDigit"};
  std::unique_ptr<TTreeReaderValue<dataformats::MCTruthContainer<MCLabel>>> mLabels{};
  bool mUseMC = true;
  bool mUseIRFrames = false;

  void connectTree(std::string filename)
  {
    auto file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
      throw std::invalid_argument(fmt::format("Opening file {} failed", filename));
    }

    auto tree = file->Get<TTree>("o2sim");
    if (!tree) {
      throw std::invalid_argument(fmt::format("Tree o2sim not found in {}", filename));
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
      pc.outputs().snapshot(OutputRef{"digits"}, std::vector<ColumnData>{});
      if (mUseMC) {
        pc.outputs().snapshot(OutputRef{"labels"}, dataformats::MCTruthContainer<MCLabel>{});
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
    std::vector<ColumnData> digits{};
    dataformats::MCTruthContainer<MCLabel> labels{};

    // get the IR frames to select
    auto irFrames = pc.inputs().get<gsl::span<dataformats::IRFrame>>("driverInfo");

    if (!irFrames.empty()) {
      utils::IRFrameSelector irfSel{};
      irfSel.setSelectedIRFrames(irFrames, 0, 0, 0, true);
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
        if (!mRofs->empty() && mRofs->front().interactionRecord <= irMax &&
            mRofs->back().interactionRecord >= irMin) {
          for (const auto& rof : *mRofs) {
            if (irfSel.check(rof.interactionRecord) != -1) {
              rofs.emplace_back(rof);
              rofs.back().firstEntry = digits.size();
              rofs.back().nEntries = rof.nEntries;
              digits.insert(digits.end(), mDigits->begin() + rof.firstEntry, mDigits->begin() + rof.getEndIndex());
              if (mUseMC) {
                for (auto idig = 0; idig < rof.nEntries; ++idig) {
                  labels.addElements(labels.getIndexedSize(), (*mLabels)->getLabels(rof.firstEntry + idig));
                }
              }
            }
          }
        }

        // move to the next TF if needed and if any
        if ((mRofs->empty() || mRofs->back().interactionRecord < irMax) &&
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

DataProcessorSpec getDigitReaderSpec(bool useMC, const char* baseDescription)
{
  std::vector<OutputSpec> outputs;
  std::stringstream ss;
  ss << "digits:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "/0";
  ss << ";rofs:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "ROF/0";
  if (useMC) {
    ss << ";labels:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "LABELS/0";
  }
  auto matchers = select(ss.str().c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    "MIDDigitsReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitsReaderDeviceDPL>(useMC)},
    Options{{"mid-digit-infile", VariantType::String, "middigits.root", {"Name of the input file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}
} // namespace mid
} // namespace o2
