// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for FT0&FV0 digits

#include "FITDigitWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/MCLabel.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TTree.h>
#include <TBranch.h>
#include <TFile.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>
#include <string>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace fit
{

class FITDPLDigitWriter
{

  using MCCont = o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>;

 public:
  void init(framework::InitContext& ic)
  {
    std::string detStrL = mID.getName();
    std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);

    auto filename = ic.options().get<std::string>((detStrL + "-digit-outfile").c_str());
    auto treename = ic.options().get<std::string>("treename");

    mOutFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
    if (!mOutFile || mOutFile->IsZombie()) {
      LOG(ERROR) << "Failed to open " << filename << " output file";
    } else {
      LOG(INFO) << "Opened " << filename << " output file";
    }
    mOutTree = std::make_unique<TTree>(treename.c_str(), treename.c_str());
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    std::string detStr = mID.getName();
    std::string detStrL = mID.getName();
    std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);

    // retrieve the digits from the input
    auto inDigits = pc.inputs().get<std::vector<o2::ft0::Digit>>((detStr + "digits").c_str());

    auto inLabels = pc.inputs().get<MCCont*>((detStr + "digitsMCTR").c_str());
    LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();

    auto digitsP = &inDigits;
    auto labelsRaw = inLabels.get();
    // connect this to a particular branch

    auto brDig = getOrMakeBranch(*mOutTree.get(), (detStr + "Digit").c_str(), &digitsP);
    auto brLbl = getOrMakeBranch(*mOutTree.get(), (detStr + "DigitMCTruth").c_str(), &labelsRaw);
    mOutTree->Fill();

    mOutFile->cd();
    mOutTree->Write();
    mOutTree.reset(); // delete the tree before closing the file
    mOutFile->Close();
    mFinished = true;
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 protected:
  FITDPLDigitWriter() = default;
  template <typename T>
  TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
  {
    if (auto br = tree.GetBranch(brname.c_str())) {
      br->SetAddress(static_cast<void*>(ptr));
      return br;
    }
    // otherwise make it
    return tree.Branch(brname.c_str(), ptr);
  }

  bool mFinished = false;
  o2::detectors::DetID mID;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;
  std::vector<o2::ft0::Digit> mDigits; // input digits
  std::unique_ptr<TFile> mOutFile;
  std::unique_ptr<TTree> mOutTree;
};

//_______________________________________________
class FT0DPLDigitWriter : public FITDPLDigitWriter
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::FT0;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginFT0;
  FT0DPLDigitWriter()
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID FT0DPLDigitWriter::DETID;
constexpr o2::header::DataOrigin FT0DPLDigitWriter::DETOR;

//_______________________________________________
/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
DataProcessorSpec getFT0DigitWriterSpec()
{
  std::string detStr = o2::detectors::DetID::getName(FT0DPLDigitWriter::DETID);
  std::string detStrL = detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto detOrig = FT0DPLDigitWriter::DETOR;

  std::vector<InputSpec> inputs;
  inputs.emplace_back(InputSpec{(detStr + "digits").c_str(), detOrig, "DIGITS", 0, Lifetime::Timeframe});
  inputs.emplace_back(InputSpec{(detStr + "digitsMCTR").c_str(), detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe});

  return DataProcessorSpec{
    (detStr + "DigitWriter").c_str(),
    inputs,
    {}, // no output
    AlgorithmSpec(adaptFromTask<FT0DPLDigitWriter>()),
    Options{
      {(detStrL + "-digit-outfile").c_str(), VariantType::String, (detStrL + "digits.root").c_str(), {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}

} // end namespace fit
} // end namespace o2
