// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCDigitRootFileWriterSpec.cxx
/// @author Matthias Richter, Sandro Wenzel
/// @since  2018-04-19
/// @brief  Processor spec for a ROOT file writer for TPC digits

#include "TPCDigitRootWriterSpec.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "CommonDataFormat/RangeReference.h"
#include "Framework/InputRecord.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/WorkflowSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/CommonMode.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;

namespace o2
{
template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace tpc
{

/// create the processor spec
/// describing a processor aggregating digits for various TPC sectors and writing them to file
/// MC truth information is also aggregated and written out
DataProcessorSpec getTPCDigitRootWriterSpec(std::vector<int> const& laneConfiguration, bool mctruth)
{
  // the callback to be set as hook for custom action when the writer is closed
  auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
    // check/verify number of entries (it should be same in all branches)

    // will return a TObjArray
    const auto brlist = outputtree->GetListOfBranches();
    int entries = -1; // init to -1 (as unitialized)
    for (TObject* entry : *brlist) {
      auto br = static_cast<TBranch*>(entry);
      int brentries = br->GetEntries();
      entries = std::max(entries, brentries);
      if (brentries != entries && !TString(br->GetName()).Contains("CommonMode")) {
        LOG(WARNING) << "INCONSISTENT NUMBER OF ENTRIES IN BRANCH " << br->GetName() << ": " << entries << " vs " << brentries;
      }
    }
    if (entries > 0) {
      LOG(INFO) << "Setting entries to " << entries;
      outputtree->SetEntries(entries);
      // outputtree->Write("", TObject::kOverwrite);
      outputfile->Close();
    }
  };

  //branch definitions for RootTreeWriter spec
  using DigitsOutputType = std::vector<o2::tpc::Digit>;
  using CommonModeOutputType = std::vector<o2::tpc::CommonMode>;

  // extracts the sector from header of an input
  auto extractSector = [](auto const& ref) {
    auto sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    if (!sectorHeader) {
      throw std::runtime_error("Missing sector header in TPC data");
    }
    // the TPCSectorHeader now allows to transport information for more than one sector,
    // e.g. for transporting clusters in one single data block. The digitization is however
    // only on sector level
    if (sectorHeader->sector() >= TPCSectorHeader::NSectors) {
      throw std::runtime_error("Digitizer can only work on single sectors");
    }
    return sectorHeader->sector();
  };

  // The generic writer needs a way to associate incoming data with the individual branches for
  // the TPC sectors. The sector number is transmitted as part of the sector header, the callback
  // finds the corresponding index in the vector of configured sectors
  auto getIndex = [laneConfiguration, extractSector](o2::framework::DataRef const& ref) -> size_t {
    auto sector = extractSector(ref);
    if (sector < 0) {
      // special data sets, don't write
      return ~(size_t)0;
    }
    size_t index = 0;
    for (auto const& s : laneConfiguration) {
      if (sector == s) {
        return index;
      }
      ++index;
    }
    throw std::runtime_error("sector " + std::to_string(sector) + " not configured for writing");
  };

  // callback to create branch name
  auto getName = [laneConfiguration](std::string base, size_t index) -> std::string {
    return base + "_" + std::to_string(laneConfiguration.at(index));
  };

  // container for cached grouping of digits
  auto trigP2Sect = std::make_shared<std::array<std::vector<DigiGroupRef>, 36>>();

  // preprocessor callback
  // read the trigger data first and store in the trigP2Sect shared pointer
  auto preprocessor = [extractSector, trigP2Sect](ProcessingContext& pc) {
    for (auto& cont : *trigP2Sect) {
      cont.clear();
    }
    std::vector<InputSpec> filter = {
      {"check", ConcreteDataTypeMatcher{"TPC", "DIGTRIGGERS"}, Lifetime::Timeframe},
    };
    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      auto sector = extractSector(ref);
      auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
      LOG(INFO) << "HAVE TRIGGER DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
      if (sector >= 0) {
        // extract the trigger information and make it available for the other handlers
        auto triggers = pc.inputs().get<std::vector<DigiGroupRef>>(ref);
        (*trigP2Sect)[sector].assign(triggers.begin(), triggers.end());
        const auto& trigS = (*trigP2Sect)[sector];
        LOG(INFO) << "GOT Triggers of sector " << sector << " | SIZE " << trigS.size();
      }
    }
  };

  // handler to fill the digit branch, this handles filling based on the trigger information, each trigger
  // will be a new entry
  auto fillDigits = [extractSector, trigP2Sect](TBranch& branch, DigitsOutputType const& digiData, DataRef const& ref) {
    auto sector = extractSector(ref);
    auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
    LOG(INFO) << "HAVE DIGIT DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
    if (sector >= 0) {
      LOG(INFO) << "DIGIT SIZE " << digiData.size();
      const auto& trigS = (*trigP2Sect.get())[sector];
      int entries = 0;
      if (!trigS.size()) {
        std::runtime_error("Digits for sector " + std::to_string(sector) + " are received w/o info on grouping in triggers");
      } else { // check consistency of Ndigits with that of expected from the trigger
        int nExp = trigS.back().getFirstEntry() + trigS.back().getEntries() - trigS.front().getFirstEntry();
        if (nExp != digiData.size()) {
          LOG(ERROR) << "Number of digits " << digiData.size() << " is inconsistent with expectation " << nExp
                     << " from digits grouping for sector " << sector;
        }
      }

      {
        if (trigS.size() == 1) { // just 1 entry (continous mode?), use digits directly
          auto ptr = &digiData;
          branch.SetAddress(&ptr);
          branch.Fill();
          entries++;
          branch.ResetAddress();
          branch.DropBaskets("all");
        } else {                                // triggered mode (>1 entries will be written)
          std::vector<o2::tpc::Digit> digGroup; // group of digits related to single trigger
          auto ptr = &digGroup;
          branch.SetAddress(&ptr);
          for (auto const& group : trigS) {
            digGroup.clear();
            for (int i = 0; i < group.getEntries(); i++) {
              digGroup.emplace_back(digiData[group.getFirstEntry() + i]); // fetch digits of given trigger
            }
            branch.Fill();
            entries++;
          }
          branch.ResetAddress();
          branch.DropBaskets("all");
        }
      }
      auto tree = branch.GetTree();
      tree->SetEntries(entries);
      tree->Write("", TObject::kOverwrite);
    }
  };

  // handler for labels
  // TODO: this is almost a copy of the above, reduce to a single methods with amends
  auto fillLabels = [extractSector, trigP2Sect](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& ref) {
    o2::dataformats::IOMCTruthContainerView outputcontainer;
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labeldata(labelbuffer);
    // first of all redefine the output format (special to labels)
    auto tree = branch.GetTree();
    auto sector = extractSector(ref);
    auto ptr = &outputcontainer;
    auto br = framework::RootTreeWriter::remapBranch(branch, &ptr);

    auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
    LOG(INFO) << "HAVE LABEL DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
    int entries = 0;
    if (sector >= 0) {
      LOG(INFO) << "MCTRUTH ELEMENTS " << labeldata.getIndexedSize()
                << " WITH " << labeldata.getNElements() << " LABELS";
      const auto& trigS = (*trigP2Sect.get())[sector];
      if (!trigS.size()) {
        throw std::runtime_error("MCTruth for sector " + std::to_string(sector) + " are received w/o info on grouping in triggers");
      } else {
        int nExp = trigS.back().getFirstEntry() + trigS.back().getEntries() - trigS.front().getFirstEntry();
        if (nExp != labeldata.getIndexedSize()) {
          LOG(ERROR) << "Number of indexed (label) slots " << labeldata.getIndexedSize()
                     << " is inconsistent with expectation " << nExp
                     << " from digits grouping for sector " << sector;
        }
      }
      {
        if (trigS.size() == 1) { // just 1 entry (continous mode?), use labels directly
          outputcontainer.adopt(labelbuffer);
          br->Fill();
          br->ResetAddress();
          br->DropBaskets("all");
          entries = 1;
        } else {
          o2::dataformats::MCTruthContainer<o2::MCCompLabel> lblGroup; // labels for group of digits related to single trigger
          for (auto const& group : trigS) {
            lblGroup.clear();
            for (int i = 0; i < group.getEntries(); i++) {
              auto lbls = labeldata.getLabels(group.getFirstEntry() + i);
              lblGroup.addElements(i, lbls);
            }
            // init the output container
            std::vector<char> flatbuffer;
            lblGroup.flatten_to(flatbuffer);
            outputcontainer.adopt(flatbuffer);
            br->Fill();
            br->DropBaskets("all");
            entries++;
          }
          br->ResetAddress();
        }
      }
      tree->SetEntries(entries);
      tree->Write("", TObject::kOverwrite);
    }
  };

  // A spectator to print logging for the common mode data
  auto commonModeSpectator = [extractSector](CommonModeOutputType const& commonModeData, DataRef const& ref) {
    auto sector = extractSector(ref);
    auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
    LOG(INFO) << "HAVE COMMON MODE DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
    LOG(INFO) << "COMMON MODE SIZE " << commonModeData.size();
  };

  auto digitsdef = BranchDefinition<DigitsOutputType>{InputSpec{"digits", ConcreteDataTypeMatcher{"TPC", "DIGITS"}},
                                                      "TPCDigit", "digits-branch-name",
                                                      laneConfiguration.size(),
                                                      fillDigits,
                                                      getIndex,
                                                      getName};

  auto labelsdef = BranchDefinition<std::vector<char>>{InputSpec{"labelinput", ConcreteDataTypeMatcher{"TPC", "DIGITSMCTR"}},
                                                       "TPCDigitMCTruth", "labels-branch-name",
                                                       // this branch definition is disabled if MC labels are not processed
                                                       (mctruth ? laneConfiguration.size() : 0),
                                                       fillLabels,
                                                       getIndex,
                                                       getName};

  auto commddef = BranchDefinition<CommonModeOutputType>{InputSpec{"commonmode", ConcreteDataTypeMatcher{"TPC", "COMMONMODE"}},
                                                         "TPCCommonMode", "common-mode-branch-name",
                                                         laneConfiguration.size(),
                                                         commonModeSpectator,
                                                         getIndex,
                                                         getName};

  return MakeRootTreeWriterSpec("TPCDigitWriter", "tpcdigits.root", "o2sim",
                                // the preprocessor reads the trigger info object and makes it available
                                // to the Fill handlers
                                MakeRootTreeWriterSpec::Preprocessor{preprocessor},
                                // defining the input for the trigger object, as an auxiliary input it is
                                // not written to any branch
                                MakeRootTreeWriterSpec::AuxInputRoute{{"triggerinput", ConcreteDataTypeMatcher{"TPC", "DIGTRIGGERS"}}},
                                // setting a custom callback for closing the writer
                                MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                // passing the branch configuration as argument pack
                                std::move(digitsdef), std::move(labelsdef), std::move(commddef))();
}
} // end namespace tpc
} // end namespace o2
