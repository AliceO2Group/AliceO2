// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include "TRDRawReaderSpec.h"

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

#include <TTree.h>
#include <TFile.h>
#include <TSystem.h>
#

using namespace o2::framework;


namespace o2
{
namespace trd
{

class TRDRawReaderSpecTask{

        public:
           void init(o2::framework::InitContext& ic)
           {
            LOG(info) << "TRD Raw Reader DPL initialising";
               LOG(info) << "Input data file is : " << ic.options().get<std::string>("simdatasrc");
            auto inputFname = ic.options().get<std::string>("simdatasrc");
            mInputFile.open(inputFname,ios::binary);
            if(!mInputFile.is_open()){
                LOG(fatal) << "Cannot open input file for TrapRawReader";
                throw invalid_argument("Cannot open input file"+inputFname);
            }
            auto stop = [this](){
                //close the input file.
                LOG(info) << " closing incoming file to raw reader in init method";
                this->mInputFile.close();
            };
            ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
           }

           void run(o2::framework::ProcessingContext &pc)
           {
               LOG(info) << "TRD Raw Reader DPL running over incoming message ...";

               auto context = pc.inputs().get<o2::steer::RunContext*>("TRD");
     //          std::vector<o2::trd::Digit> digits;
      //         o4::dataformat::MCTruthContainer<o2::trd::MCLabel> labels;
               LOG(info) << "and we are in the run method of TRD RawReader DPL \\o/ ";
               cout << ".... and we are in the run method ....." << endl;

           }
       private:
           TrapSimulator mTrapSimulator;
           std::ifstream mInputFile{};  // incoming data file for digits and MC labels.

};



o2::framework::DataProcessorSpec getTRDRawReaderSpec() 
{
    return DataProcessorSpec{"TRDIN",Inputs{},
                                     Outputs{OutputSpec{"TRD", "DIGITS", 0, Lifetime::Timeframe},
                                             OutputSpec{"TRD", "LABELS", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptFromTask<TRDRawReaderSpecTask>()},
                                    Options{
                                        {"simdatasrc",VariantType::String,"run2digitmanager",{"Input data file containing run2 dump of digitsmanager going into MCMSim"}}
                                    }
                            };
}

} //end namespace trd
} //end namespace o2
