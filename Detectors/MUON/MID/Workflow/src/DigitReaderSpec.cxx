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
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2019

#include "MIDWorkflow/DigitReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"
#include "TFile.h"
#include "TTree.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC = true) : mUseMC(useMC) {}

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
    mTree->SetBranchAddress("MIDDigit", &mDigits);
    if (mUseMC) {
      mTree->SetBranchAddress("MIDDigitMCLabels", &mMCContainer);
    }
    mTree->SetBranchAddress("MIDROFRecords", &mROFRecords);
    mState = 0;
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mState != 0) {
      return;
    }

    std::vector<ColumnData> digits;
    o2::dataformats::MCTruthContainer<MCLabel> mcContainer;
    std::vector<ROFRecord> rofRecords;

    for (auto ientry = 0; ientry < mTree->GetEntries(); ++ientry) {
      mTree->GetEntry(ientry);
      digits.insert(digits.end(), mDigits->begin(), mDigits->end());
      rofRecords.insert(rofRecords.end(), mROFRecords->begin(), mROFRecords->end());
      if (mUseMC) {
        mcContainer.mergeAtBack(*mMCContainer);
      }
    }

    LOG(DEBUG) << "MIDDigitsReader pushed " << digits.size() << " merged digits";
    pc.outputs().snapshot(of::Output{"MID", "DATA", 0, of::Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(of::Output{"MID", "DATAROF", 0, of::Lifetime::Timeframe}, rofRecords);
    LOG(DEBUG) << "MIDDigitsReader pushed " << digits.size() << " indexed digits";
    if (mUseMC) {
      pc.outputs().snapshot(of::Output{"MID", "DATALABELS", 0, of::Lifetime::Timeframe}, mcContainer);
    }
    mState = 2;
    pc.services().get<of::ControlService>().endOfStream();
  }

 private:
  std::unique_ptr<TFile> mFile{nullptr};
  TTree* mTree{nullptr};                                             // not owner
  std::vector<o2::mid::ColumnDataMC>* mDigits{nullptr};              // not owner
  o2::dataformats::MCTruthContainer<MCLabel>* mMCContainer{nullptr}; // not owner
  std::vector<o2::mid::ROFRecord>* mROFRecords{nullptr};             // not owner
  int mState = 0;
  bool mUseMC = true;
};

framework::DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<of::OutputSpec> outputs;
  outputs.emplace_back("MID", "DATA", o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("MID", "DATAROF", o2::framework::Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("MID", "DATALABELS", o2::framework::Lifetime::Timeframe);
  }

  return of::DataProcessorSpec{
    "MIDDigitsReader",
    of::Inputs{},
    outputs,
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::DigitsReaderDeviceDPL>(useMC)},
    of::Options{
      {"mid-digit-infile", of::VariantType::String, "middigits.root", {"Name of the input file"}}}};
}
} // namespace mid
} // namespace o2
