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
#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapSimulator.h"

using namespace o2::framework;


namespace o2
{
namespace trd
{

class TRDDPLTrapSimulatorTask{

        public:
           void init(o2::framework::InitContext& ic)
           {
//               LOG(info) << "Input data file is : " << ic.options().get<std::string>("simdatasrc");
//               LOG(info) << "simSm is : " << ic.options().get<int>("simSM");
            mfeeparam=FeeParam::instance();
            LOG(info) << "TRD Trap Simulator DPL initialising with pid of : "<< ::getpid() << "and feeparam pointer is " << hex << mfeeparam;
             
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
               auto digits = pc.inputs().get<std::vector<o2::trd::Digit> >("digitinput");
               auto mclabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::trd::MCLabel>* >("labelinput");
               LOG(info) << "digits size is : "<< digits.size();
              // now loop over the digits for a given trap.
              // send to trapsimulator 
              // repeat 
              //mTrapSimulator.init();
              ArrayADC_t incomingdigits;
              // std::array<unsigned short,30> mcmdigits; //TODO come back and pull timebins from somehwere.
               for (auto digit : digits) {
                      int pad=digit.getPad();
                      int row=digit.getRow();
                      int detector=digit.getDetector();
                      LOG(info) << "MCM: " <<  mfeeparam->getMCMfromPad(row, pad) << " == "<< detector <<"::"<<row<<"::"<< pad;
                      int rob=mfeeparam->getROBfromPad(row,pad);
                      int mcm=mfeeparam->getMCMfromPad(row,pad);
                      // copy adc data from digits to local array and then pass into TrapSimulator
                      //mTrapSimulator.init(detector,rob,mcm);
                      //mTrapSimulator.setData(detector,digit.getADC());
             //for(int i=i;i<digits.size();i++){
             //   int mcmindex= FeeParam::instance()->getMCMfromPad(digits[i].getRow(), digits[i].getPad());
                //      LOG(info) << "MCM: " <<  feeparam->getMCMfromPad(digits[i].getRow(),digits[i].getPad()) << " == "<< digits[i].getDetector() <<"::"<<digits[i].getRow()<<"::"<< digits[i].getPad();
                      //cout << "pad: " << ((o2::trd::Digit)digits[i]).getPad() << endl;
               //       LOG(info) << "MCM: " << i <<" :: " << mcmindex << " == "<< digits[i].getDetector() <<"::"<<digits[i].getRow()<<"::"<< digits[i].getPad();
                       
              }
               
               LOG(info) << "and we are in the run method of TRDDPLTrapSimulatorTask \\o/ ";
           //    cout << ".... and we are in the run method ....." << endl;

           }
       private:
           TrapSimulator mTrapSimulator;
           FeeParam *mfeeparam;
};



o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(int channelfan) 
{
    return DataProcessorSpec{"TRAP",Inputs{ InputSpec{"digitinput","TRD","DIGITS",0},
                                            InputSpec{"labelinput","TRD","LABELS",0}
                                          },
                                    Outputs{OutputSpec{"TRD","TRACKLETS",0}},
                                          AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()}};
}

} //end namespace trd
} //end namespace o2
