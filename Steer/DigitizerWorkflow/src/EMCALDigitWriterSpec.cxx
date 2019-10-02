// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for EMCAL digits

#include "EMCALDigitWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TBranch.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace emcal
{

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

void DigitsWriterSpec::init(framework::InitContext& ctx)
{
  // get the option from the init context
  auto filename = ctx.options().get<std::string>("emcal-digit-outfile");
  auto treename = ctx.options().get<std::string>("treename");
  mOutputFile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
  mOutputTree = std::make_shared<TTree>(treename.c_str(), treename.c_str());
  mDigits = std::make_shared<std::vector<o2::emcal::Digit>>();
  mFinished = false;

  // the callback to be set as hook at stop of processing for the framework
  auto outputfile = mOutputFile;
  auto outputtree = mOutputTree;
  auto finishWriting = [outputfile, outputtree]() {
    outputtree->SetEntries(1);
    outputtree->Write();
    outputfile->Close();
  };
  ctx.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);
}

void DigitsWriterSpec::run(framework::ProcessingContext& ctx)
{
  if (mFinished)
    return;

  // retrieve the digits from the input
  auto indata = ctx.inputs().get<std::vector<o2::emcal::Digit>>("emcaldigits");
  LOG(INFO) << "RECEIVED DIGITS SIZE " << indata.size();
  *mDigits.get() = std::move(indata);

  // connect this to a particular branch
  auto br = getOrMakeBranch(*mOutputTree.get(), "EMCALDigit", mDigits.get());
  br->Fill();

  // retrieve labels from the input
  auto labeldata = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("emcaldigitlabels");
  LOG(INFO) << "EMCAL GOT " << labeldata->getNElements() << " LABELS ";
  auto labeldataraw = labeldata.get();
  // connect this to a particular branch
  auto labelbr = getOrMakeBranch(*mOutputTree.get(), "EMCALDigitMCTruth", &labeldataraw);
  labelbr->Fill();

  mFinished = true;
  ctx.services().get<ControlService>().readyToQuit(false);
}

/// create the processor spec
/// describing a processor receiving digits for EMCal writing them to file
DataProcessorSpec getEMCALDigitWriterSpec()
{
  return DataProcessorSpec{
    "EMCALDigitWriter",
    Inputs{InputSpec{"emcaldigits", "EMC", "DIGITS", 0, Lifetime::Timeframe},
           InputSpec{"emcaldigitlabels", "EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}},
    {}, // no output
    AlgorithmSpec(framework::adaptFromTask<DigitsWriterSpec>()),
    Options{
      {"emcal-digit-outfile", VariantType::String, "emcaldigits.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}
} // end namespace emcal
} // end namespace o2
