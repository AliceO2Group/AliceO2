// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include "TRDDigitReaderSpec.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h> // for getppid


#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "DPLUtils/RootTreeReader.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "TRDBase/Digit.h" // for the Digit type
#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h" // for the Hit type
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsTRD/TriggerRecord.h"
#include <TTree.h>
#include <TFile.h>
#include <TSystem.h>

using namespace o2::framework;



namespace o2
{
namespace trd
{

class TRDDigitReaderSpecTask{

        public:
           void init(o2::framework::InitContext& ic)
           {
            LOG(info) << "TRD Raw Reader DPL initialising";
               LOG(info) << "Input data file is : " << ic.options().get<std::string>("simdatasrc");
            auto inputFname = ic.options().get<std::string>("simdatasrc");
            mFile = std::make_unique<TFile>(inputFname.c_str());
            if(!mFile.IsOpen()){
                LOG(error) << "Cannot open input file for DigitReader";
                mState=1;
                return;
            }
    mTree = static_cast<TTree*>(mFile->Get("o2sim"));
    if (!mTree) {
      LOG(ERROR) << "Cannot find tree in " << inputFname;
      mState = 1;
      return;
    }
    mTree->SetBranchAddress("TRDDigit", &mDigit);
    mTree->SetBranchAddress("TRDMCLabels", &mMCLabels);
    mTree->SetBranchAddress("TriggerRecord", &mTriggerlRecords);
    mState = 0;
  }


           void run(o2::framework::ProcessingContext &pc)
           {
               LOG(info) << "TRD Digit Reader DPL running over incoming message ...";

//               auto context = pc.inputs().get<o2::steer::RunContext*>("TRD");
     //          std::vector<o2::trd::Digit> digits;
      //         o4::dataformat::MCTruthContainer<o2::trd::MCLabel> labels;
               LOG(info) << "and we are in the run method of TRD RawReader DPL \\o/ ";
               cout << ".... and we are in the run method ....." << endl;
            if (mState != 0) {
                return;
            }

    std::vector<o2::trd:Digit> digits;
    o2::dataformats::MCTruthContainer<MCLabel> mcLabels;
    std::vector<o2::trd::TriggerRecords> TriggerRecords;

    for (auto ientry = 0; ientry < mTree->GetEntries(); ++ientry) {
      mTree->GetEntry(ientry);
      std::copy(mDigits->begin(), mDigits->end(), std::back_inserter(digits));
      mcLabels.mergeAtBack(*mMCLabels);
      std::copy(mTriggerRecords->begin(), mTriggerRecords->end(), std::back_inserter(TriggerRecords));
    }

    pc.outputs().snapshot(of::Output{"TRD", "DIGITS", 0, of::Lifetime::Timeframe}, mDigits);
    LOG(DEBUG) << "TRDDigitsReader pushed  merged digits";
    pc.outputs().snapshot(of::Output{"TRD", "TRGLABELS", 0, of::Lifetime::Timeframe}, mMCLabels);
    LOG(DEBUG) << "TRDDigitsReader pushed triggerlabels";
    pc.outputs().snapshot(of::Output{"TRD", "LABELS", 0, of::Lifetime::Timeframe}, mTriggerRecords);
    LOG(DEBUG) << "TRDDigitsReader pushed MC labels";

    mState = 2;

           }
       private:
           TrapSimulator mTrapSimulator;
           std::unique_ptr<TFile> mFile{nullptr};
           TTree*  mDigits{nullptr};
           std::vector<o2::trd::Digit>* mDigits{nullptr};              // not owner
           o2::dataformats::MCTruthContainer<MCLabel>* mMCLabels{nullptr}; // not owner
           std::vector<o2::trd::TriggerRecord>* mTriggerRecords{nullptr};             // not owner
           int mState=0; // a mechanism to abort the run method if the files are not setup properly or opened or other.

};



o2::framework::DataProcessorSpec getTRDDigitReaderSpec() 
{
    return DataProcessorSpec{"TRDIN",Inputs{},
                                     Outputs{OutputSpec{"TRD", "DIGITS", 0, Lifetime::Timeframe},
                                             OutputSpec{"TRD", "LABELS", 0, Lifetime::Timeframe}},
                                             OutputSpec{"TRD", "TRGLABELS", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptFromTask<TRDDigitReaderSpecTask>()},
                                    Options{
                                    //    {"simdatasrc2",VariantType::String,"run2digitmanager",{"Input data file containing run2 dump of digitsmanager going into Trap Simulator"}},
                                       {"simdatasrc",VariantType::String,"trddigits.root",{"Input data file containing run3 digitizer going into Trap Simulator"}}
                                   }
                            };
}

} //end namespace trd
} //end namespace o2
