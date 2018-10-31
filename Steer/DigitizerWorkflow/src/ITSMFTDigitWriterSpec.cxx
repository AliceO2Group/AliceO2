// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for ITSMFT digits

#include "ITSMFTDigitWriterSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "ITSMFTBase/Digit.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/ROFRecord.h"
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
namespace ITSMFT
{

class ITSMFTDPLDigitWriter
{

  using MCCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

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
    auto inDigits = pc.inputs().get<std::vector<o2::ITSMFT::Digit>>((detStr + "digits").c_str());
    auto inROFs = pc.inputs().get<std::vector<o2::ITSMFT::ROFRecord>>((detStr + "digitsROF").c_str());
    auto inMC2ROFs = pc.inputs().get<std::vector<o2::ITSMFT::MC2ROFRecord>>((detStr + "digitsMC2ROF").c_str());
    auto inLabels = pc.inputs().get<MCCont*>((detStr + "digitsMCTR").c_str());
    LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();

    auto digitsP = &inDigits;
    auto labelsRaw = inLabels.get();
    // connect this to a particular branch

    auto brDig = getOrMakeBranch(*mOutTree.get(), (detStr + "Digit").c_str(), &digitsP);
    auto brLbl = getOrMakeBranch(*mOutTree.get(), (detStr + "DigitMCTruth").c_str(), &labelsRaw);
    mOutTree->Fill();

    mOutFile->cd();
    mOutFile->WriteObjectAny(&inROFs, "std::vector<o2::ITSMFT::ROFRecord>", (detStr + "DigitROF").c_str());
    mOutFile->WriteObjectAny(&inMC2ROFs, "std::vector<o2::ITSMFT::MC2ROFRecord>", (detStr + "DigitMC2ROF").c_str());
    mOutTree->Write();
    mOutTree.reset(); // delete the tree before closing the file
    mOutFile->Close();
    mFinished = true;
    pc.services().get<ControlService>().readyToQuit(false);
  }

 protected:
  ITSMFTDPLDigitWriter() {}
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
  std::vector<o2::ITSMFT::Digit> mDigits; // input digits
  std::unique_ptr<TFile> mOutFile;
  std::unique_ptr<TTree> mOutTree;
};

//_______________________________________________
class ITSDPLDigitWriter : public ITSMFTDPLDigitWriter
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::ITS;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginITS;
  ITSDPLDigitWriter()
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID ITSDPLDigitWriter::DETID;
constexpr o2::header::DataOrigin ITSDPLDigitWriter::DETOR;

//_______________________________________________
class MFTDPLDigitWriter : public ITSMFTDPLDigitWriter
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::MFT;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginMFT;
  MFTDPLDigitWriter()
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID MFTDPLDigitWriter::DETID;
constexpr o2::header::DataOrigin MFTDPLDigitWriter::DETOR;

/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
DataProcessorSpec getITSDigitWriterSpec()
{
  std::string detStr = o2::detectors::DetID::getName(ITSDPLDigitWriter::DETID);
  std::string detStrL = detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto detOrig = ITSDPLDigitWriter::DETOR;

  std::vector<InputSpec> inputs;
  inputs.emplace_back(InputSpec{ (detStr + "digits").c_str(), detOrig, "DIGITS", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsROF").c_str(), detOrig, "DIGITSROF", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsMC2ROF").c_str(), detOrig, "DIGITSMC2ROF", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsMCTR").c_str(), detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe });

  return DataProcessorSpec{
    (detStr + "DigitWriter").c_str(),
    inputs,
    {}, // no output
    AlgorithmSpec(adaptFromTask<ITSDPLDigitWriter>()),
    Options{
      { (detStrL + "-digit-outfile").c_str(), VariantType::String, (detStrL + "digits.root").c_str(), { "Name of the input file" } },
      { "treename", VariantType::String, "o2sim", { "Name of top-level TTree" } },
    }
  };
}

DataProcessorSpec getMFTDigitWriterSpec()
{
  std::string detStr = o2::detectors::DetID::getName(MFTDPLDigitWriter::DETID);
  std::string detStrL = detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto detOrig = MFTDPLDigitWriter::DETOR;

  std::vector<InputSpec> inputs;
  inputs.emplace_back(InputSpec{ (detStr + "digits").c_str(), detOrig, "DIGITS", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsROF").c_str(), detOrig, "DIGITSROF", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsMC2ROF").c_str(), detOrig, "DIGITSMC2ROF", 0, Lifetime::Timeframe });
  inputs.emplace_back(InputSpec{ (detStr + "digitsMCTR").c_str(), detOrig, "DIGITSMCTR", 0, Lifetime::Timeframe });
  return DataProcessorSpec{
    (detStr + "DigitWriter").c_str(),
    inputs,
    {}, // no output
    AlgorithmSpec(adaptFromTask<MFTDPLDigitWriter>()),
    Options{
      { (detStrL + "-digit-outfile").c_str(), VariantType::String, (detStrL + "digits.root").c_str(), { "Name of the input file" } },
      { "treename", VariantType::String, "o2sim", { "Name of top-level TTree" } },
    }
  };
}

} // end namespace ITSMFT
} // end namespace o2
