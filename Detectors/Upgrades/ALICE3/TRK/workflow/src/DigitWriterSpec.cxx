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

/// @brief  Processor spec for a ROOT file writer for TRK digits (per-layer)

#include "TRKWorkflow/DigitWriterSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataRef.h"
#include "TRKBase/AlmiraParam.h"
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
#include <format>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace trk
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MCCont = o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>;

DataProcessorSpec getTRKDigitWriterSpec(bool mctruth, bool dec, bool calib)
{
  static constexpr o2::header::DataOrigin Origin = o2::header::gDataOriginTRK;
  const int mLayers = o2::trk::AlmiraParam::kNLayers;
  std::string detStr = "TRK";
  std::string detStrL = dec ? "o2_trk" : "trk";

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
    outputfile->Write("", TObject::kOverwrite);
    outputfile->Close();
  };

  // handler for labels
  auto fillLabels = [detStr, digitSizes, rofSizes](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& ref) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto layer = static_cast<size_t>(dh->subSpecification);
    LOG(info) << detStr << ": WRITING " << labels.getNElements() << " LABELS"
              << std::format(" FOR LAYER {}", layer) << " WITH " << (*digitSizes)[layer]
              << " DIGITS IN " << (*rofSizes)[layer] << " ROFS";

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
    return base + "_" + std::to_string(index);
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

  return MakeRootTreeWriterSpec(("TRKDigitWriter" + std::string(dec ? "_dec" : "")).c_str(),
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
                                BranchDefinition<std::vector<itsmft::GBTCalibData>>{InputSpec{detStr + "calib", ConcreteDataTypeMatcher{Origin, "GBTCALIB"}},
                                                                                    detStr + "Calib", "digit-calib-branch",
                                                                                    (calib ? 1 : 0)})();
}

} // end namespace trk
} // end namespace o2
