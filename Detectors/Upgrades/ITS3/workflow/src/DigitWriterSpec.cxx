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

#include "ITS3Workflow/DigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <array>
#include <string>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace its3
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MCCont = o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>;

/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
DataProcessorSpec getDigitWriterSpec(bool mctruth, bool dec, bool calib, o2::header::DataOrigin detOrig, o2::detectors::DetID detId)
{
  std::string detStr = o2::detectors::DetID::getName(detId);
  std::string detStrL = dec ? "o2_" : ""; // for decoded digits prepend by o2
  detStrL += detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);

  auto digitSizes = std::make_shared<std::vector<size_t>>(7, 0);
  auto digitSizeGetter = [digitSizes](std::vector<o2::itsmft::Digit> const& inDigits, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    (*digitSizes)[dh->subSpecification] = inDigits.size();
  };
  auto rofSizes = std::make_shared<std::vector<size_t>>(7, 0);
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
    outputtree->Write("", TObject::kOverwrite);
    outputfile->Close();
  };

  // handler for labels
  // This is necessary since we can't store the original label buffer in a ROOT entry -- as is -- if it exceeds a certain size.
  // We therefore convert it to a special split class.
  auto fillLabels = [detStr, digitSizes, rofSizes](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& ref) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto layer = static_cast<size_t>(dh->subSpecification);
    LOG(info) << detStr << ": WRITING " << labels.getNElements() << " LABELS FOR LAYER " << layer << " WITH " << (*digitSizes)[layer] << " DIGITS IN " << (*rofSizes)[layer] << " ROFS";

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
  auto getName = [](std::string base, size_t index) -> std::string {
    return base += "_" + std::to_string(index);
  };

  std::vector<InputSpec> vecInpSpecDig, vecInpSpecROF, vecInpSpecLbl;
  vecInpSpecDig.reserve(7);
  vecInpSpecROF.reserve(7);
  vecInpSpecLbl.reserve(7);
  for (int iLayer = 0; iLayer < 7; iLayer++) {
    vecInpSpecDig.emplace_back(getName("digits", iLayer), "IT3", "DIGITS", iLayer);
    vecInpSpecROF.emplace_back(getName("digitsROF", iLayer), "IT3", "DIGITSROF", iLayer);
    vecInpSpecLbl.emplace_back(getName("digitsMCTR", iLayer), "IT3", "DIGITSMCTR", iLayer);
  }

  return MakeRootTreeWriterSpec((detStr + "DigitWriter" + (dec ? "_dec" : "")).c_str(),
                                (detStrL + "digits.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Digits tree"},
                                MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                // in case of labels we first read them as std::vector<char> and process them correctly in the fillLabels hook
                                BranchDefinition<std::vector<itsmft::Digit>>{vecInpSpecDig,
                                                                             detStr + "Digit", "digit-branch",
                                                                             7,
                                                                             digitSizeGetter,
                                                                             getIndex,
                                                                             getName},
                                BranchDefinition<std::vector<itsmft::ROFRecord>>{vecInpSpecROF,
                                                                                 detStr + "DigitROF", "digit-rof-branch",
                                                                                 7,
                                                                                 rofSizeGetter,
                                                                                 getIndex,
                                                                                 getName},
                                BranchDefinition<std::vector<char>>{vecInpSpecLbl,
                                                                    "IT3DigitMCTruth", "digit-mctruth-branch",
                                                                    (mctruth ? 7 : 0),
                                                                    fillLabels,
                                                                    getIndex,
                                                                    getName})();
}

DataProcessorSpec getITS3DigitWriterSpec(bool mctruth, bool dec, bool calib)
{
  return getDigitWriterSpec(mctruth, dec, calib, o2::header::gDataOriginIT3, o2::detectors::DetID::IT3);
}

} // end namespace its3
} // end namespace o2
