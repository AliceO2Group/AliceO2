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

/// @brief  Processor spec for a ROOT file writer for ITSMFT digits

#include "ITSMFTWorkflow/DigitWriterSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataRef.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace itsmft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MCCont = o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>;

/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
template <int N>
DataProcessorSpec getDigitWriterSpec(bool mctruth, bool doStag, bool dec, bool calib)
{
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  int mLayers = doStag ? o2::itsmft::DPLAlpideParam<N>::getNLayers() : 1;
  std::string detStr = o2::detectors::DetID::getName(N);
  std::string detStrL = dec ? "o2_" : ""; // for decoded digits prepend by o2
  detStrL += detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto digitSizes = std::make_shared<std::vector<size_t>>(mLayers, 0);
  auto digitSizeGetter = [digitSizes](std::vector<o2::itsmft::Digit> const& inDigits, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    (*digitSizes)[dh->subSpecification] = inDigits.size();
  };
  auto rofSizes = std::make_shared<std::vector<size_t>>(mLayers, 0);
  auto rofSizeGetter = [rofSizes](std::vector<o2::itsmft::ROFRecord> const& inROFs, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    (*rofSizes)[dh->subSpecification] = inROFs.size();
  };

  // the callback to be set as hook for custom action when the writer is closed
  auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
    const auto* brArr = outputtree->GetListOfBranches();
    int64_t nent = 0;
    for (const auto* brc : *brArr) {
      int64_t n = ((const TBranch*)brc)->GetEntries();
      if (nent && (nent != n)) {
        LOG(error) << "Branches have different number of entries";
      }
      nent = n;
    }
    outputtree->SetEntries(nent);
    // do not use TTree::Write .. as this writes to default directory (not the associated file)
    // instead of outputtree->Write("", TObject::kOverwrite)
    // --> better use TFile::Write or TFile::WriteObject
    outputfile->Write("", TObject::kOverwrite);
    outputfile->Close();
  };

  // handler for labels
  // This is necessary since we can't store the original label buffer in a ROOT entry -- as is -- if it exceeds a certain size.
  // We therefore convert it to a special split class.
  auto fillLabels = [detStr, doStag, digitSizes, rofSizes](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& ref) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto layer = static_cast<size_t>(dh->subSpecification);
    LOG(info) << detStr << ": WRITING " << labels.getNElements() << " LABELS" << (doStag ? std::format(" FOR LAYER {}", layer) : "") << " WITH " << (*digitSizes)[layer] << " DIGITS IN " << (*rofSizes)[layer] << " ROFS";

    o2::dataformats::IOMCTruthContainerView outputcontainer;
    auto ptr = &outputcontainer;
    auto br = framework::RootTreeWriter::remapBranch(branch, &ptr);
    outputcontainer.adopt(labelbuffer);
    br->Fill();
    br->ResetAddress();
  };

  auto getIndex = [](DataRef const& ref) -> size_t {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    return static_cast<size_t>(dh->subSpecification);
  };
  auto getName = [doStag](std::string base, size_t index) -> std::string {
    if (doStag) {
      return base += "_" + std::to_string(index);
    }
    return base;
  };

  std::vector<InputSpec> vecInpSpecDig, vecInpSpecROF, vecInpSpecLbl;
  vecInpSpecDig.reserve(mLayers);
  vecInpSpecROF.reserve(mLayers);
  vecInpSpecLbl.reserve(mLayers);
  for (int iLayer = 0; iLayer < mLayers; iLayer++) {
    vecInpSpecDig.emplace_back(getName(detStr + "digits", iLayer), Origin, "DIGITS", iLayer);
    vecInpSpecROF.emplace_back(getName(detStr + "digitsROF", iLayer), Origin, "DIGITSROF", iLayer);
    vecInpSpecLbl.emplace_back(getName(detStr + "_digitsMCTR", iLayer), Origin, "DIGITSMCTR", iLayer);
  }

  return MakeRootTreeWriterSpec((detStr + "DigitWriter" + (dec ? "_dec" : "")).c_str(),
                                (detStrL + "digits.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{.name = "o2sim", .title = detStr + " Digits tree"},
                                MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                BranchDefinition<std::vector<itsmft::Digit>>{vecInpSpecDig,
                                                                             detStr + "Digit", "digit-branch",
                                                                             mLayers,
                                                                             digitSizeGetter,
                                                                             getIndex,
                                                                             getName},
                                BranchDefinition<std::vector<itsmft::ROFRecord>>{vecInpSpecROF,
                                                                                 detStr + "DigitROF", "digit-rof-branch",
                                                                                 mLayers,
                                                                                 rofSizeGetter,
                                                                                 getIndex,
                                                                                 getName},
                                BranchDefinition<std::vector<char>>{vecInpSpecLbl,
                                                                    detStr + "DigitMCTruth", "digit-mctruth-branch",
                                                                    (mctruth ? mLayers : 0),
                                                                    fillLabels,
                                                                    getIndex,
                                                                    getName},
                                BranchDefinition<std::vector<itsmft::MC2ROFRecord>>{InputSpec{detStr + "_digitsMC2ROF", ConcreteDataTypeMatcher{Origin, "DIGITSMC2ROF"}},
                                                                                    detStr + "DigitMC2ROF", "digit-mc2rof-branch",
                                                                                    (mctruth ? mLayers : 0),
                                                                                    getIndex,
                                                                                    getName},
                                BranchDefinition<std::vector<itsmft::GBTCalibData>>{InputSpec{detStr + "calib", ConcreteDataTypeMatcher{Origin, "GBTCALIB"}},
                                                                                    detStr + "Calib", "digit-calib-branch",
                                                                                    (calib ? 1 : 0)})();
}

DataProcessorSpec getITSDigitWriterSpec(bool mctruth, bool doStag, bool dec, bool calib)
{
  return getDigitWriterSpec<o2::detectors::DetID::ITS>(mctruth, doStag, dec, calib);
}

DataProcessorSpec getMFTDigitWriterSpec(bool mctruth, bool doStag, bool dec, bool calib)
{
  return getDigitWriterSpec<o2::detectors::DetID::MFT>(mctruth, doStag, dec, calib);
}

} // end namespace itsmft
} // end namespace o2
