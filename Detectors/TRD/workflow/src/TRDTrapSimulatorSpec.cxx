// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


//#include "Framework/runDataProcessing.h"
#include "TRDTrapSimulatorSpec.h"

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

using namespace o2::framework;


namespace o2
{
namespace trd
{

class TRDDPLTrapSimulatorTask{

        public:
           void init(o2::framework::InitContext& ic)
           {
            LOG(info) << "TRD Trap Simulator DPL initialising";

           }
           void run(o2::framework::ProcessingContext &pc)
           {
               LOG(info) << "TRD Trap Simulator DPL running over incoming message ...";
               // basic idea, just for my sanity.
               //get message
               //unpack by mcm, should be sequential on mcm.
               //call trap simulator for each mcm in the message and do its thing.
               //package the resultant tracklets into the outgoing message.
               //send outgoing message.

               auto context = pc.inputs().get<o2::steer::RunContext*>("TRD");
     //          std::vector<o2::trd::Digit> digits;
      //         o4::dataformat::MCTruthContainer<o2::trd::MCLabel> labels;
               LOG(info) << "and we are in the run method of TRDDPLTrapSimulatorTask \\o/ ";
               cout << ".... and we are in the run method ....." << endl;

           }
       private:
           TrapSimulator mTrapSimulator;
};



/* void customize(std::vector<o2::framework::ConfigParamSpec>& workflowoptions)
{
    //able to specify inputs
    //could be disk, upstream digitizer, run2 convert.
    //most of these are probably purely for debugging.
    //specify where the data is coming from i.e. ignore incoming message and use data as specified here, mostly for debugging as well.
    std::string trapsimindatahelp("Specify the location of incoming data for the simulator, full name of file");
    workflowoptions.push_back(ConfigParamSpec{"tsimdatasrc", VariantType::String, "none", {trapsimindatahelp}});

    //limit the trapsim to a specific roc or multiple rocs mostly for debugging.
    std::string trapsimrochelp("Specify the ROC to work on [0-540]");
    workflowoptions.push_back(ConfigParamSpec{"tsimroc", VariantType::Int, "none", {trapsimrochelp}});

    //limit to 1 supermodule.
    std::string trapsimsupermodulehelp("Specify the Supermodule to work on [0-18]");
    workflowoptions.push_back(ConfigParamSpec{"tsimSM", VariantType::Int, "none", {trapsimsupermodulehelp}});
    //limit to a stack in a supermodule
    std::string trapsimstackhelp("Specify the specific stack to work on [0-5] within the supermodule");
    workflowoptions.push_back(ConfigParamSpec{"tsimstack", VariantType::Int, "none", {trapsimstackhelp}});
    
    //probably more options to come.
}
*/
//#include "Framework/runDataProcessing.h"


o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(int channelfan) 
{
    return DataProcessorSpec{"TRAP",Inputs{ InputSpec{"digtinput","TRD","DIGITS",0},
                                            InputSpec{"labelinput","TRD","LABELS",0}
                                          },
                                    Outputs{OutputSpec{"TRD","TRACK",0}},
                                          AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()}};
}

} //end namespace trd
} //end namespace o2
