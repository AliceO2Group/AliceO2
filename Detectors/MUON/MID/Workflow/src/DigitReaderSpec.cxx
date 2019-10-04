// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MID digits reader device
/// \author Diego Stocco <dstocco at cern.ch>
/// \date   11 April 2019

#include "MIDWorkflow/DigitReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDSimulation/DigitsPacker.h"
#include "MIDSimulation/MCLabel.h"
#include "TFile.h"
#include "TTree.h"

#include <fairlogger/Logger.h>

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class DigitsReaderDeviceDPL
{
 public:
  void init(o2::framework::InitContext& ic)
  {
    auto filename = ic.options().get<std::string>("mid-digit-infile");
    mFile = std::make_unique<TFile>(filename.c_str());
    if (!mFile->IsOpen()) {
      LOG(ERROR) << "Cannot open the " << filename << " file !";
      mState = 1;
      return;
    }
    mTree = static_cast<TTree*>(mFile->Get("o2sim"));
    if (!mTree) {
      LOG(ERROR) << "Cannot find tree in " << filename;
      mState = 1;
      return;
    }
    mTimediff = ic.options().get<int>("mid-digit-timediff");
    mTree->SetBranchAddress("MIDDigit", &mDigits);
    mTree->SetBranchAddress("MIDDigitMCLabels", &mMCContainer);
    mState = 0;
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mState != 0) {
      return;
    }

    if (!nextGroup()) {
      pc.services().get<of::ControlService>().readyToQuit(framework::QuitRequest::Me);
      return;
    }

    std::vector<o2::mid::ColumnDataMC> packedDigits;
    std::vector<o2::mid::ColumnData> data;
    o2::dataformats::MCTruthContainer<MCLabel> packedMCContainer, outMCContainer;

    mDigitsPacker.getGroup(mCurrentGroup, packedDigits, packedMCContainer);
    LOG(INFO) << "MIDDigitsReader found " << packedDigits.size() << " unmerged MC digits with same timestamp";
    mDigitsMerger.process(packedDigits, packedMCContainer, data, outMCContainer);
    LOG(INFO) << "MIDDigitsReader pushed " << data.size() << " merged digits";
    pc.outputs().snapshot(of::Output{"MID", "DATA", 0, of::Lifetime::Timeframe}, data);
    LOG(INFO) << "MIDDigitsReader pushed " << outMCContainer.getIndexedSize() << " indexed digits";
    pc.outputs().snapshot(of::Output{"MID", "DATALABELS", 0, of::Lifetime::Timeframe}, outMCContainer);
  }

 private:
  bool nextGroup()
  {
    if (mCurrentEntry == mTree->GetEntries()) {
      return false;
    }
    ++mCurrentGroup;
    if (mCurrentEntry < 0 || mCurrentGroup == mDigitsPacker.getNGroups()) {
      ++mCurrentEntry;
      if (mCurrentEntry == mTree->GetEntries()) {
        return false;
      }
      mTree->GetEntry(mCurrentEntry);
      mCurrentGroup = 0;
    }
    mDigitsPacker.process(*mDigits, *mMCContainer, mTimediff);
    return true;
  }

  DigitsMerger mDigitsMerger;
  DigitsPacker mDigitsPacker;
  std::unique_ptr<TFile> mFile = nullptr;
  TTree* mTree = nullptr;                                             // not owner
  std::vector<o2::mid::ColumnDataMC>* mDigits = nullptr;              // not owner
  o2::dataformats::MCTruthContainer<MCLabel>* mMCContainer = nullptr; // not owner
  int mTimediff = 0;
  int mState = 0;
  int mCurrentGroup = -1;
  int mCurrentEntry = -1;
};

framework::DataProcessorSpec getDigitReaderSpec()
{
  return of::DataProcessorSpec{
    "MIDDigitsReader",
    of::Inputs{},
    of::Outputs{of::OutputSpec{"MID", "DATA"}, of::OutputSpec{"MID", "DATALABELS"}},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::DigitsReaderDeviceDPL>()},
    of::Options{
      {"mid-digit-infile", of::VariantType::String, "middigits.root", {"Name of the input file"}},
      {"mid-digit-timediff", of::VariantType::Int, 0, {"Maximum difference between digits"}}}};
}
} // namespace mid
} // namespace o2
