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
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Digit.h"
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

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace TPC
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
DataProcessorSpec getTPCDigitRootWriterSpec(int numberofsourcedevices)
{
  // assign input names to each channel
  auto digitchannelname = std::make_shared<std::vector<std::string>>();
  auto labelchannelname = std::make_shared<std::vector<std::string>>();
  for (int i = 0; i < numberofsourcedevices; ++i) {
    {
      std::stringstream ss;
      ss << "digitinput" << i;
      digitchannelname->push_back(ss.str());
    }
    {
      std::stringstream ss;
      ss << "labelinput" << i;
      labelchannelname->push_back(ss.str());
    }
  }

  auto initFunction = [numberofsourcedevices, digitchannelname, labelchannelname](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("tpc-digit-outfile");
    auto treename = ic.options().get<std::string>("treename");

    auto outputfile = std::make_shared<TFile>(filename.c_str(), "RECREATE");
    auto outputtree = std::make_shared<TTree>(treename.c_str(), treename.c_str());

    // container for incoming digits + label
    auto digits = std::make_shared<std::vector<o2::TPC::Digit>>();
    auto labels = std::make_shared<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();

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
          if (brentries != entries) {
            LOG(WARNING) << "INCONSISTENT NUMBER OF ENTRIES IN BRANCHES " << entries << " vs " << brentries;
            entries = brentries;
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
    auto processingFct = [outputfile, outputtree, digits, digitchannelname, labelchannelname,
                          numberofsourcedevices](ProcessingContext& pc) {
      static bool finished = false;
      if (finished) {
        // avoid being executed again when marked as finished;
        return;
      }

      static int finishchecksum = 0;
      static int invocation = 0;
      invocation++;
      // need to record which channel has completed in order to decide when we can shutdown
      static std::vector<bool> digitsdone;
      static std::vector<bool> labelsdone;
      if (invocation == 1) {
        digitsdone.resize(numberofsourcedevices, false);
        labelsdone.resize(numberofsourcedevices, false);
      }

      auto isComplete = [numberofsourcedevices](int i) {
        if (i == numberofsourcedevices * (numberofsourcedevices - 1) / 2) {
          return true;
        }
        return false;
      };

      // extracts the sector from header of an input
      auto extractSector = [&pc](const char* inputname) {
        auto sectorHeader = DataRefUtils::getHeader<o2::TPC::TPCSectorHeader*>(pc.inputs().get(inputname));
        if (!sectorHeader) {
          LOG(FATAL) << "Missing sector header in TPC data";
        }
        return sectorHeader->sector;
      };

      int sector = -1; // TPC sector for which data was received
      for (int d = 0; d < numberofsourcedevices; ++d) {
        const auto dname = digitchannelname->operator[](d);
        const auto lname = labelchannelname->operator[](d);

        // probe which channel has data and of what kind
        // a) probe for digits:
        if (pc.inputs().isValid(dname.c_str())) {
          sector = extractSector(dname.c_str());
          LOG(INFO) << "HAVE DIGIT DATA FOR SECTOR " << sector << " ON CHANNEL " << d;
          if (sector == -1) {
            digitsdone[d] = true;
          } else {
            // have to do work ...
            // the digits
            auto indata = pc.inputs().get<std::vector<o2::TPC::Digit>>(dname.c_str());
            LOG(INFO) << "DIGIT SIZE " << indata.size();
            *digits.get() = std::move(indata);
            {
              // connect this to a particular branch
              auto br = getOrMakeBranch(*outputtree.get(), "TPCDigit", sector, digits.get());
              br->Fill();
              br->ResetAddress();
            }
          }
        } // end digit case

        // b) probe for labels
        if (pc.inputs().isValid(lname.c_str())) {
          sector = extractSector(lname.c_str());
          LOG(INFO) << "HAVE LABEL DATA FOR SECTOR " << sector << " ON CHANNEL " << d;
          if (sector == -1) {
            labelsdone[d] = true;
          } else {
            // the labels
            auto labeldata = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>(lname.c_str());
            auto labeldataRaw = labeldata.get();
            LOG(INFO) << "MCTRUTH ELEMENTS " << labeldataRaw->getIndexedSize()
                      << " WITH " << labeldataRaw->getNElements() << " LABELS ";
            if (labeldataRaw->getIndexedSize() != digits->size()) {
              LOG(WARNING) << "Inconsistent number of indexed (label) slots "
                           << labeldataRaw->getIndexedSize() << " versus digits " << digits->size();
            }
            {
              auto br = getOrMakeBranch(*outputtree.get(), "TPCDigitMCTruth", sector, &labeldataRaw);
              br->Fill();
              br->ResetAddress();
            }
          }
        } // end label case

        if (labelsdone[d] && digitsdone[d]) {
          LOG(INFO) << "CHANNEL " << d << " DONE ";

          // we must increase the checksum only once
          // prevent this by invalidating ...
          labelsdone[d] = false;
          digitsdone[d] = false;

          finishchecksum += d;
          if (isComplete(finishchecksum)) {
            finished = true;
            pc.services().get<ControlService>().readyToQuit(false);
            return;
          }
        }
      }
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  std::vector<InputSpec> inputs;
  for (int d = 0; d < numberofsourcedevices; ++d) {
    inputs.emplace_back(InputSpec{ (*digitchannelname.get())[d].c_str(), "TPC", "DIGITS",
                                   static_cast<SubSpecificationType>(d), Lifetime::Timeframe }); // digit input
    inputs.emplace_back(InputSpec{ (*labelchannelname.get())[d].c_str(), "TPC", "DIGITSMCTR",
                                   static_cast<SubSpecificationType>(d), Lifetime::Timeframe });
  }

  return DataProcessorSpec{
    "TPCDigitWriter",
    inputs,
    {}, // no output
    AlgorithmSpec(initFunction),
    Options{
      { "tpc-digit-outfile", VariantType::String, "tpcdigits.root", { "Name of the input file" } },
      { "treename", VariantType::String, "o2sim", { "Name of tree for tracks" } },
    }
  };
}
} // end namespace TPC
} // end namespace o2
