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

#include "ITSMFTWorkflow/DigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>
#include <string>
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#endif

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace itsmft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MCCont = o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>;

// #define CUSTOM 1
// make a std::vec use a gsl::span as internal buffer without copy
template <typename T>
void adopt(gsl::span<const T> const& data, std::vector<T>& v)
{
  static_assert(sizeof(v) == 24);
  if (data.size() == 0) {
    return;
  }
  // we assume a standard layout of begin, end, end_capacity and overwrite the internal members of the vector
  struct Impl {
    const T* start;
    const T* end;
    const T* cap;
  };

  Impl impl;
  impl.start = &(data[0]);
  impl.end = &(data[data.size() - 1]) + 1; // end pointer (beyond last element)
  impl.cap = impl.end;
  std::memcpy(&v, &impl, sizeof(Impl));
  assert(data.size() == v.size());
  assert(v.capacity() == v.size());
  assert((void*)&data[0] == (void*)&v[0]);
}

/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
DataProcessorSpec getDigitWriterSpec(bool mctruth, o2::header::DataOrigin detOrig, o2::detectors::DetID detId)
{
  std::string detStr = o2::detectors::DetID::getName(detId);
  std::string detStrL = detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
#ifdef CUSTOM
  std::vector<o2::framework::InputSpec> inputs;
  if (mctruth) {
    inputs.emplace_back(InputSpec{"digitsMCTR", detOrig, "DIGITSMCTR", 0});
    inputs.emplace_back(InputSpec{"digitsMC2ROF", detOrig, "DIGITSMC2ROF", 0});
  }
  inputs.emplace_back(InputSpec{"digits", detOrig, "DIGITS", 0});
  inputs.emplace_back(InputSpec{"digitsROF", detOrig, "DIGITSROF", 0});

  return {(detStr + "DigitWriter").c_str(),
          inputs,
          {},
          AlgorithmSpec{
            [detStrL, detStr, mctruth](ProcessingContext& ctx) {
              static bool mFinished = false;
              if (mFinished) {
                return;
              }

              TFile f((detStrL + "digits.root").c_str(), "RECREATE");
              TTree t("o2sim", "o2sim");
              // define data
              auto digits = new std::vector<itsmft::Digit>; // needs to be a pointer since the message memory is managed by DPL and we have to avoid double deletes
              auto rof = new std::vector<itsmft::ROFRecord>;
              auto mc2rofrecords = new std::vector<itsmft::MC2ROFRecord>;
              o2::dataformats::IOMCTruthContainerView labelview;

              auto fillBranch = [](TBranch* br) {
                br->Fill();
                br->ResetAddress();
                br->DropBaskets("all");
              };
              // get the data as gsl::span so that ideally no copy is made
              // but immediately adopt the views in standard std::vectors *** THIS IS AN INTERNAL HACK ***
              if (mctruth) {
                labelview.adopt(ctx.inputs().get<gsl::span<char>>("digitsMCTR"));
                fillBranch(t.Branch((detStr + "DigitMCTruth").c_str(), &labelview));
                adopt(ctx.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("digitsMC2ROF"), *mc2rofrecords);
                fillBranch(t.Branch((detStr + "DigitMC2ROF").c_str(), &mc2rofrecords));
              }
              adopt(ctx.inputs().get<gsl::span<itsmft::Digit>>("digits"), *digits);
              fillBranch(t.Branch((detStr + "Digits").c_str(), &digits));
              adopt(ctx.inputs().get<gsl::span<itsmft::ROFRecord>>("digitsROF"), *rof);
              fillBranch(t.Branch((detStr + "DigitROF").c_str(), &rof));
              t.SetEntries(1);
              f.Write();
              f.Close();
              ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
              mFinished = true;
            }}};
#else
  auto logger = [](std::vector<o2::itsmft::Digit> const& inDigits) {
    LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();
  };

  // the callback to be set as hook for custom action when the writer is closed
  auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
    outputtree->SetEntries(1);
    outputtree->Write("", TObject::kOverwrite);
    outputfile->Close();
  };

  // handler for labels
  // This is necessary since we can't store the original label buffer in a ROOT entry -- as is -- if it exceeds a certain size.
  // We therefore convert it to a special split class.
  auto fillLabels = [detStr](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& /*ref*/) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    LOG(INFO) << "WRITING " << labels.getNElements() << " LABELS ";

    o2::dataformats::IOMCTruthContainerView outputcontainer;
    // first of all redefine the output format (special to labels)
    auto tree = branch.GetTree();

    std::stringstream str;
    str << detStr + "DigitMCTruth";
    auto br = tree->Branch(str.str().c_str(), &outputcontainer);
    outputcontainer.adopt(labelbuffer);
    br->Fill();
    br->ResetAddress();
    const int entries = 1;
    tree->SetEntries(entries);
    tree->Write("", TObject::kOverwrite);
  };

  return MakeRootTreeWriterSpec((detStr + "DigitWriter").c_str(),
                                (detStrL + "digits.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Digits tree"},
                                MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                // in case of labels we first read them as std::vector<char> and process them correctly in the fillLabels hook
                                BranchDefinition<std::vector<char>>{InputSpec{"digitsMCTR", detOrig, "DIGITSMCTR", 0},
                                                                    (detStr + "DigitMCTruth_TMP").c_str(),
                                                                    (mctruth ? 1 : 0), fillLabels},
                                BranchDefinition<std::vector<itsmft::MC2ROFRecord>>{InputSpec{"digitsMC2ROF", detOrig, "DIGITSMC2ROF", 0},
                                                                                    (detStr + "DigitMC2ROF").c_str(),
                                                                                    (mctruth ? 1 : 0)},
                                BranchDefinition<std::vector<itsmft::Digit>>{InputSpec{"digits", detOrig, "DIGITS", 0},
                                                                             (detStr + "Digit").c_str(),
                                                                             logger},
                                BranchDefinition<std::vector<itsmft::ROFRecord>>{InputSpec{"digitsROF", detOrig, "DIGITSROF", 0},
                                                                                 (detStr + "DigitROF").c_str()})();
#endif
}

DataProcessorSpec getITSDigitWriterSpec(bool mctruth)
{
  return getDigitWriterSpec(mctruth, o2::header::gDataOriginITS, o2::detectors::DetID::ITS);
}

DataProcessorSpec getMFTDigitWriterSpec(bool mctruth)
{
  return getDigitWriterSpec(mctruth, o2::header::gDataOriginMFT, o2::detectors::DetID::MFT);
}

} // end namespace itsmft
} // end namespace o2
