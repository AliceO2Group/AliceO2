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
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/InputRecord.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/WorkflowSpec.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/CommonMode.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <memory> // for make_shared, make_unique, unique_ptr
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <gsl/gsl>

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;

namespace o2
{
namespace tpc
{

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string basename, int sector, T* ptr)
{
  std::stringstream stream;
  stream << basename << "_" << sector;
  const auto brname = stream.str();
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

/// create the processor spec
/// describing a processor aggregating digits for various TPC sectors and writing them to file
/// MC truth information is also aggregated and written out
DataProcessorSpec getTPCDigitRootWriterSpec(std::vector<int> const& laneConfiguration, bool mctruth)
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tpc-digit-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for cached grouping of digits
    // TODO: once InputRecord::ReturnType helper has been merged
    // using DigitGroupRefInputContainer = InputRecord::ReturnType<std::vector<DigiGroupRef>>;
    using DigitGroupRefInputContainer = decltype(std::declval<InputRecord>().get<std::vector<DigiGroupRef>>(DataRef{nullptr, nullptr, nullptr}));
    auto trigP2Sect = std::make_shared<std::array<DigitGroupRefInputContainer, 36>>();

    // the callback to be set as hook at stop of processing for the framework
    auto finishWriting = [outputfile, outputtree]() {
      // check/verify number of entries (it should be same in all branches)

      // will return a TObjArray
      const auto brlist = outputtree->GetListOfBranches();
      int entries = -1; // init to -1 (as unitialized)
      for (TObject* entry : *brlist) {
        auto br = static_cast<TBranch*>(entry);
        int brentries = br->GetEntries();
        if (entries == -1) {
          entries = brentries;
        } else {
          if (brentries != entries && !TString(br->GetName()).Contains("CommonMode")) {
            LOG(WARNING) << "INCONSISTENT NUMBER OF ENTRIES IN BRANCH " << br->GetName() << ": " << entries << " vs " << brentries;
          }
        }
      }
      if (entries > 0) {
        outputtree->SetEntries(entries);
      }
      outputtree->Write();
      outputfile->Close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    auto processingFct = [outputfile, outputtree, trigP2Sect](ProcessingContext& pc) {
      // extracts the sector from header of an input
      auto extractSector = [&pc](auto const& ref) {
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

      // read the trigger data first
      {
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGTRIGGERS"}, Lifetime::Timeframe},
        };
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto sector = extractSector(ref);
          auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
          LOG(INFO) << "HAVE TRIGGER DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
          if (sector >= 0) {
            auto triggers = pc.inputs().get<std::vector<DigiGroupRef>>(ref);
            (*trigP2Sect.get())[sector] = std::move(triggers);
            const auto& trigS = (*trigP2Sect.get())[sector];
            LOG(INFO) << "GOT Triggers of sector " << sector << " | SIZE " << trigS.size();
          }
        }
      }

      {
        // probe which channel has data and of what kind
        // a) probe for digits:
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe},
        };
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto sector = extractSector(ref);
          auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
          LOG(INFO) << "HAVE DIGIT DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
          if (sector >= 0) {
            // the digits
            auto digiData = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
            LOG(INFO) << "DIGIT SIZE " << digiData.size();
            const auto& trigS = (*trigP2Sect.get())[sector];
            if (!trigS.size()) {
              LOG(FATAL) << "Digits for sector " << sector << " are received w/o info on grouping in triggers";
            } else { // check consistency of Ndigits with that of expected from the trigger
              int nExp = trigS.back().getFirstEntry() + trigS.back().getEntries() - trigS.front().getFirstEntry();
              if (nExp != digiData.size()) {
                LOG(ERROR) << "Number of digits " << digiData.size() << " is inconsistent with expectation " << nExp
                           << " from digits grouping for sector " << sector;
              }
            }

            {
              if (trigS.size() == 1) { // just 1 entry (continous mode?), use digits directly
                // connect this to a particular branch
                // the input data span is directly using the raw buffer, we need to copy to
                // the object we want to write, maybe we can avoid this with some tricks
                std::vector<o2::tpc::Digit> writeObj(digiData.begin(), digiData.end());
                auto digP = &writeObj;
                auto br = getOrMakeBranch(*outputtree.get(), "TPCDigit", sector, digP);
                br->Fill();
                br->ResetAddress();
              } else {                                // triggered mode (>1 entries will be written)
                std::vector<o2::tpc::Digit> digGroup; // group of digits related to single trigger
                auto digGroupPtr = &digGroup;
                auto br = getOrMakeBranch(*outputtree.get(), "TPCDigit", sector, digGroupPtr);
                for (auto grp : trigS) {
                  digGroup.clear();
                  for (int i = 0; i < grp.getEntries(); i++) {
                    digGroup.emplace_back(digiData[grp.getFirstEntry() + i]); // fetch digits of given trigger
                  }
                  br->Fill();
                }
                br->ResetAddress();
              }
            }
          }
        } // end digit case
      }

      {
        // b) probe for labels
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe},
        };
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto sector = extractSector(ref);
          auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
          LOG(INFO) << "HAVE LABEL DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
          if (sector >= 0) {
            // the labels
            auto labeldata = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>(ref);
            auto labeldataRaw = labeldata.get();
            LOG(INFO) << "MCTRUTH ELEMENTS " << labeldataRaw->getIndexedSize()
                      << " WITH " << labeldataRaw->getNElements() << " LABELS";
            const auto& trigS = (*trigP2Sect.get())[sector];
            if (!trigS.size()) {
              LOG(FATAL) << "MCTruth for sector " << sector << " are received w/o info on grouping in triggers";
            } else {
              int nExp = trigS.back().getFirstEntry() + trigS.back().getEntries() - trigS.front().getFirstEntry();
              if (nExp != labeldataRaw->getIndexedSize()) {
                LOG(ERROR) << "Number of indexed (label) slots " << labeldataRaw->getIndexedSize()
                           << " is inconsistent with expectation " << nExp
                           << " from digits grouping for sector " << sector;
              }
            }
            {
              if (trigS.size() == 1) { // just 1 entry (continous mode?), use labels directly
                auto br = getOrMakeBranch(*outputtree.get(), "TPCDigitMCTruth", sector, &labeldataRaw);
                br->Fill();
                br->ResetAddress();
              } else {
                o2::dataformats::MCTruthContainer<o2::MCCompLabel> lblGroup; // labels for group of digits related to single trigger
                auto lblGroupPtr = &lblGroup;
                auto br = getOrMakeBranch(*outputtree.get(), "TPCDigitMCTruth", sector, &lblGroupPtr);
                for (auto grp : trigS) {
                  lblGroup.clear();
                  for (int i = 0; i < grp.getEntries(); i++) {
                    auto lbls = labeldataRaw->getLabels(grp.getFirstEntry() + i);
                    lblGroup.addElements(i, lbls);
                  }
                  br->Fill();
                }
              }
            }
          }
        } // end label case
      }

      {
        // c) probe for common mode:
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "COMMONMODE"}, Lifetime::Timeframe},
        };
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto sector = extractSector(ref);
          auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
          LOG(INFO) << "HAVE COMMON MODE DATA FOR SECTOR " << sector << " ON CHANNEL " << dh->subSpecification;
          const auto& trigS = (*trigP2Sect.get())[sector];
          if (sector >= 0) {
            auto commonModeData = pc.inputs().get<std::vector<o2::tpc::CommonMode>>(ref);
            LOG(INFO) << "COMMON MODE SIZE " << commonModeData.size();

            if (!trigS.size()) {
              LOG(FATAL) << "CommonMode for sector " << sector << " are received w/o info on grouping in triggers";
            }
            {
              auto digC = &commonModeData;
              auto br = getOrMakeBranch(*outputtree.get(), "TPCCommonMode", sector, digC);
              br->Fill();
              br->ResetAddress();
            }
          }
        } // end common mode case
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs = {
    {"digitinput", "TPC", "DIGITS", 0, Lifetime::Timeframe},        // digit input
    {"triggerinput", "TPC", "DIGTRIGGERS", 0, Lifetime::Timeframe}, // groupping in triggers
    {"commonmodeinput", "TPC", "COMMONMODE", 0, Lifetime::Timeframe},
  };
  if (mctruth) {
    inputs.emplace_back("labelinput", "TPC", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  auto amendInput = [&laneConfiguration](InputSpec& spec, size_t index) {
    spec.binding += std::to_string(laneConfiguration[index]);
    DataSpecUtils::updateMatchingSubspec(spec, laneConfiguration[index]);
  };
  inputs = mergeInputs(inputs, laneConfiguration.size(), amendInput);

  return DataProcessorSpec{
    "TPCDigitWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      {"tpc-digit-outfile", VariantType::String, "tpcdigits.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of tree for tracks"}},
    }};
}
} // end namespace tpc
} // end namespace o2
