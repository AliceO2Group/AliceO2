// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCDigitMergerWriterSpec.cxx
/// @author Sandro Wenzel
/// @since  2021-03-10
/// @brief  Processor spec for post-step; merging exisisting ROOT files
///         into a single one

#include "TPCDigitMergerWriterSpec.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/CommonMode.h"
#include "DetectorsBase/Detector.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

template <typename T>
void copyHelper(T const& origin, T& target)
{
  std::copy(origin.begin(), origin.end(), std::back_inserter(target));
}
template <>
void copyHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& origin, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& target)
{
  target.mergeAtBack(origin);
}

template <typename T>
void writeToBranchHelper(TTree& tree, const char* name, T* accum)
{
  auto targetbr = o2::base::getOrMakeBranch(tree, name, accum);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}

template <>
void writeToBranchHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(TTree& tree,
                                                                             const char* name, o2::dataformats::MCTruthContainer<o2::MCCompLabel>* accum)
{
  // we convert first of all to IOMCTruthContainer
  std::vector<char> buffer;
  accum->flatten_to(buffer);
  accum->clear_andfreememory();
  o2::dataformats::IOMCTruthContainerView view(buffer);
  auto targetbr = o2::base::getOrMakeBranch(tree, name, &view);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}

// just produce a final chain on existing files
// void produceChain(std::vector<int> const& lanes) {
// auto newfile = new TFile("tpcdigits.root", "RECREATE");
//  assert(newfile);
//  auto chain = new TChain("o2sim", "o2sim");
// assert(newtree);
// for(auto l : lanes) {
// merging the data
//  std::stringstream tmp;
//  tmp << "tpcdigits_lane" << l << ".root";
//  auto originfile = new TFile(tmp.str().c_str(), "OPEN");
//  assert(originfile);
// }
//}

void produceMergedTimeframeFile(std::vector<int> const& lanes)
{
  //
  auto newfile = new TFile("tpcdigits.root", "RECREATE");
  assert(newfile);
  auto newtree = new TTree("o2sim", "o2sim");
  assert(newtree);
  for (auto l : lanes) {
    LOG(INFO) << "MERGING FOR LANE " << l;
    // merging the data
    std::stringstream tmp;
    tmp << "tpc_driftime_digits_lane" << l << ".root";
    auto originfile = new TFile(tmp.str().c_str(), "OPEN");
    assert(originfile);

    auto merge = [originfile, newfile, newtree](auto data, auto brprefix) {
      auto keyslist = originfile->GetListOfKeys();
      for (int i = 0; i < keyslist->GetEntries(); ++i) {
        auto key = keyslist->At(i);
        auto oldtree = (TTree*)originfile->Get(key->GetName());
        assert(oldtree);
        std::stringstream digitbrname;
        digitbrname << brprefix << key->GetName();
        auto br = oldtree->GetBranch(digitbrname.str().c_str());
        if (!br) {
          continue;
        }
        decltype(data)* chunk = nullptr;
        br->SetAddress(&chunk);
        decltype(data) accum;
        for (auto e = 0; e < br->GetEntries(); ++e) {
          br->GetEntry(e);
          copyHelper(*chunk, accum);
          delete chunk;
          chunk = nullptr;
        }
        br->ResetAddress();
        br->DropBaskets("all");
        writeToBranchHelper(*newtree, br->GetName(), &accum);
        newfile->Write("", TObject::kOverwrite);
        delete oldtree;
      }
    };

    //data definitions
    using DigitsType = std::vector<o2::tpc::Digit>;
    using CommonModeType = std::vector<o2::tpc::CommonMode>;
    using LabelType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    merge(DigitsType(), "TPCDigit_");
    merge(LabelType(), "TPCDigitMCTruth_");
    merge(CommonModeType(), "TPCCommonMode_");
    originfile->Close();
    delete originfile;
  }
  newfile->Close();
  delete newfile;
}

/// create the processor spec
/// describing a processor aggregating digits for various TPC sectors and writing them to file
/// MC truth information is also aggregated and written out
DataProcessorSpec getTPCDigitMergerWriterSpec(std::vector<int> const& laneConfiguration, bool mctruth)
{
  //data definitions
  using DigitsOutputType = std::vector<o2::tpc::Digit>;
  using CommonModeOutputType = std::vector<o2::tpc::CommonMode>;
}

} // end namespace tpc
} // end namespace o2

int main(int argc, char* argv[])
{
  int numlanes = atoi(argv[1]);
  std::vector<int> lanes(numlanes);
  std::iota(lanes.begin(), lanes.end(), 0);
  o2::tpc::produceMergedTimeframeFile(lanes);
}
