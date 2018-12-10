// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx

#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/DigitReaderSpec.h"
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

DataProcessorSpec getDigitReaderSpec()
{
  auto init = [](InitContext &ic) {
    auto filename = ic.options().get<std::string>("its-digit-infile");
	  
    return [filename](ProcessingContext &pc) {
      static bool done=false;
      if (done) return;

      TFile file(filename.c_str(),"OLD");
      if (file.IsOpen()) {
	 std::unique_ptr<TTree> tree((TTree*)file.Get("o2sim"));
	 if (tree) {
	   std::vector<o2::ITSMFT::Digit> allDigits;
	   std::vector<o2::ITSMFT::Digit> digits, *pdigits=&digits;
	   tree->SetBranchAddress("ITSDigit", &pdigits);
	   o2::dataformats::MCTruthContainer<o2::MCCompLabel> allLabels;
	   o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels, *plabels=&labels;
	   tree->SetBranchAddress("ITSDigitMCTruth", &plabels);

	   int ne = tree->GetEntries();
           for (int e=0; e<ne; e++) {
	     tree->GetEntry(e);
	     std::copy(digits.begin(), digits.end(), std::back_inserter(allDigits));
	     allLabels.mergeAtBack(labels);
	   }
           LOG(INFO)<<"ITSDigitReader pushed "<<allDigits.size()<<" digits";
           pc.outputs().snapshot(Output{"ITS","DIGITS",     0, Lifetime::Timeframe}, allDigits);
           pc.outputs().snapshot(Output{"ITS","DIGITSMCTR", 0, Lifetime::Timeframe}, allLabels);
	 } else {
	   LOG(ERROR)<<"Cannot find the o2sim tree !";
	 }
      } else {
	LOG(ERROR)<<"Cannot open the "<< filename.c_str()<<" file !";
      }
      done=true;
      //pc.services().get<ControlService>().readyToQuit(true);
    };
  };

  return DataProcessorSpec {
    "its-digit-reader",
    Inputs{},
    Outputs{
      OutputSpec{"ITS", "DIGITS",     0, Lifetime::Timeframe},
      OutputSpec{"ITS", "DIGITSMCTR", 0, Lifetime::Timeframe}
    },
    AlgorithmSpec{ init },
    Options{
       { "its-digit-infile", VariantType::String, "o2digi_its.root", { "Name of the input file" } }
    }
  };
}
  
}
}
