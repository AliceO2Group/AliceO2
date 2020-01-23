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


bool DigitSortComparator( o2::trd::Digit const& a, o2::trd::Digit const &b)
{
int roba,robb,mcma,mcmb;
int rowa,rowb,pada,padb;
FeeParam *fee=FeeParam::instance();
double timea,timeb;
rowa=a.getRow();rowb=b.getRow();
pada=a.getPad();padb=b.getPad();
timea=a.getTime();timeb=b.getTime();
roba=fee->getROBfromPad(rowa,pada);
robb=fee->getROBfromPad(rowb,padb);
mcma=fee->getMCMfromPad(rowa,pada);
mcmb=fee->getMCMfromPad(rowb,padb);
if(timea < timeb) return 1;
else if (timea==timeb){

  if(a.getDetector()<b.getDetector()) return 1;
  else {
   if(a.getDetector()==b.getDetector()){
      if(roba<robb) return 1;
      else{ 
          if (roba==robb){
              if(mcma<mcmb) return 1;
              else return 0;
          }    
          else return 0;
          }
      return 0;
      }
   return 0;
   }
  return 0;
  }
return 0;
}

bool DigitSortComparatorPadRow( o2::trd::Digit const& a, o2::trd::Digit const &b)
{
//this sorts the data into pad rows, and pads with in that pad row.
//this allows us to generate a structure of 144 pads to then pass to the trap simulator 
//taking into account the shared pads.
int roba,robb,mcma,mcmb;
int rowa,rowb,pada,padb;
FeeParam *fee=FeeParam::instance();
double timea,timeb;
rowa=a.getRow();rowb=b.getRow();
pada=a.getPad();padb=b.getPad();
timea=a.getTime();timeb=b.getTime();
roba=fee->getROBfromPad(rowa,pada);
robb=fee->getROBfromPad(rowb,padb);
mcma=fee->getMCMfromPad(rowa,pada);
mcmb=fee->getMCMfromPad(rowb,padb);
if(timea < timeb) return 1;
else if (timea==timeb){

  if(a.getDetector()<b.getDetector()) return 1;
  else {
   if(a.getDetector()==b.getDetector()){
      if(rowa<rowb) return 1;
      else{ 
            if (rowa==rowb){
                if(pada<padb) return 1; //leaving in for now. so I dont have to have a 144x30 entry array for the pad data. dont have to deal with pad sorting, its going to be 1 of 144 and inserted into the required array.
              else return 0;
          }    
          else return 0;
          }
      return 0;
      }
   return 0;
   }
  return 0;
  }
return 0;
}







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
               auto digits = pc.inputs().get<std::vector<o2::trd::Digit> >("digitinput");
               auto mclabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::trd::MCLabel>* >("labelinput");
               LOG(info) << "digits size is : "<< digits.size();
              
              //first sort digits into detector::padrow order
              std::sort(digits.begin(),digits.end(), DigitSortComparator);//sortRuleLambda );
              //TODO optimisation : sort by frequency of trap chip and do all instances of the same trap at each init.
              //TODO dont sort, just build an index array 
                   int oldmcm=-1;
                   int oldrob=-1;
                   int olddetector=-1;
                   int oldrow=-1;
                   int oldpad=-1;
                   unsigned char firedtraps=0;                   
                   for(std::vector<o2::trd::Digit>::iterator digititerator = digits.begin(); digititerator != digits.end(); digititerator++) {
                      //originally loop was over side:rob:mcm so we need side
                      //in here we have an entire padrow which corresponds to 8 MCM.
                      //while on a single padrow, populate array padsinrow.
                      //on change of padrow
                      //  fireup trapsim, do its thing with each 18 sequence of pads by copying relevant data into the 20 ADCs of the mcm as well. as per the end of AliTRDdigitizer
                      double digittime=digititerator->getTime();
                      int pad=digititerator->getPad();
                      int row=digititerator->getRow();
                      int detector=digititerator->getDetector();
                      int rob=mfeeparam->getROBfromPad(row,pad);
                      int mcm=mfeeparam->getMCMfromPad(row,pad);
                      if(oldmcm==-1) oldmcm=mcm;
                      if(oldrob==-1) oldrob=rob;
                      if(oldpad==-1) oldpad=pad;
                      if(oldrow==-1) oldrow=pad;
                      if(olddetector==-1) olddetector=detector;
                      //LOG(info) << "Det : " << detector << " Row : " << row << " pad : " << pad << " rob: "<< rob << " mcm: "<< mcm << " first mcm : " << mfeeparam->getMCMfromPad(row,0) << " last mcm : " << mfeeparam->getMCMfromPad(row,143) << " diff : " << 
                      //    mfeeparam->getMCMfromPad(row,143)- mfeeparam->getMCMfromPad(row, 2) << "test of calc " << (143%(144/2))/18+4*(row%4)  << " test of first pad : " << (1%(144/2))/18+4*(row%4); 
                      //if(i%18==0 || (i-1)%18==0 || (i+1)%18==0 ) LOG(info) << "shared row : " << row << " pad : " << i << " robfromshared:" << mfeeparam->getROBfromSharedPad(row,i) << " mcmfromshared:" << mfeeparam->getMCMfromSharedPad(row,i);
                      //determine which adc for this mcm we are populating.

                      if(olddetector!=detector || oldrow!= row){
                        //all data is inside the 8 relavent trapsimulators
                        //mcm=mfeeparam->getMCMfromPad(oldrow,oldpad);

                        rob=mfeeparam->getROBfromPad(oldrow,oldpad);// 
                        LOG(debug1) << "processing of row,mcm" << " padrow changed from " << olddetector <<"," << oldrow << " to " << detector << "," << row;
                        //fireup Trapsim.
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "*********************************  BEGIN *****************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "PROCESSING THE ARRAY OF TRAP CHIPS !";
                        for(int trapcounter=0;trapcounter<8;trapcounter++){
                            unsigned int isinit=mTrapSimulator[trapcounter].checkInitialized();
                            LOG(debug4) << "is init : " << isinit ;
                            if(mTrapSimulator[trapcounter].isDataSet()){//firedtraps & (1<<trapcounter){ /}/mTrapSimulator[trapcounter].checkInitialized()){
                            LOG(debug1) << "["<<trapcounter << "] PROCESSING TRAP !";
                                //this one has been filled with data for the now previous pad row.
                                //Well the first question is, have it *actually* been filled with data ....
                                //
                                LOG(debug1) << "************************* trap data for trap : " << trapcounter;
                            //    mTrapSimulator[trapcounter].printAdcDatHuman(cout);
                                mTrapSimulator[trapcounter].filter();
                                if(mfeeparam->getTracklet()){
                                    mTrapSimulator[trapcounter].tracklet();
                                    mTrapSimulator[trapcounter].storeTracklets();
                                }
                                mTrapSimulator[trapcounter].zeroSupressionMapping();
                                //TODO build message.
                                //
                                //last step is to clear this trap chip for the next round.
                                //mTrapSimulator[trapcounter].reset();// reset will get called at the end of the next init call.
                                //set this trap sim object to have not data (effectively) reset.
                                mTrapSimulator[trapcounter].unsetData();
                            }
                            else {
                                LOG(debug4) << "if statement is init failed ";
                                LOG(debug1) << "["<<trapcounter << "] PROCESSING TRAP !";
                            }
                        }
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "*************************************** END **************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        LOG(debug1) << "**********************************************************************************************";
                        // mTrapSimulator.init(detector,rob,mcm);
                        // mTrapSimulator.setData(detector,digit.getADC());

                        //now clean up 

                        
                      }
                      else {
                          //we are still on the same detector and row.
                          //add the digits to the padrow.
                          //copy pad time data into where they belong in the 8 TrapSimulators for this pad.
                          int mcmoffset=-1;
                          int firstrob=mfeeparam->getROBfromPad(row,5); // 5 is arbitrary, but above the lower shared pads. so will get first rob and mcm
                          int firstmcm=mfeeparam->getMCMfromPad(row,5); // 5 for same reason
                          int trapindex=pad/18; 

                          //check trap is initialised.
                          if(!mTrapSimulator[trapindex].checkInitialized()){
                              LOG(debug4) << "Initialised trapsimulator for triplet ("<< detector << "," << rob << "," << mcm << ") as its not initialized and we need to send it some adc data.";
                              mTrapSimulator[trapindex].init(detector,rob,mcm);
                              LOG(debug4) << "Initialised trapsimulator for triplet ("<< detector << "," << rob << "," << mcm << ") as its not initialized and we need to send it some adc data.";
                          }
                          int adc=0; 
                          adc=20-(pad % 18)-1;
                              LOG(debug1) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!................................... setting data for simulator : " << trapindex << " and adc : " << adc;
                          mTrapSimulator[trapindex].setData(adc,digititerator->getADC());
                         // mTrapSimulator[trapindex].printAdcDatHuman(cout);
                          //LOG(debug3) << "pad: " << pad << " robfromPad:" << mfeeparam->getROBfromPad(row,pad) << " mcmfrompad:" << mfeeparam->getMCMfromPad(row,pad) << " adc : " << adc;
                          // now take care of the case of shared pads (the whole reason for doing this pad row wise).

                          if(pad%18==0 || (pad+1)%18==0){ //case of pad 18 and 19 must be shared to preceding trap chip adc 1 and 0 respectively.
                          //check trap is initialised.
                          
                              adc=20-(pad % 18)-1;
                              LOG(debug1) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.................................... setting data for simulator : " << trapindex-1 << " and adc : " << adc;
                              mTrapSimulator[trapindex-1].setData(adc,digititerator->getADC());
                         // mTrapSimulator[trapindex].printAdcDatHuman(cout);
                      //        LOG(debug3) << "shared pad: " << pad << " pad : " << pad << " robfromshared:" << mfeeparam->getROBfromSharedPad(row,pad) << " mcmfromshared:" << mfeeparam->getMCMfromSharedPad(row,pad) << " adc : " << adc;
                          } 
                          if((pad-1)%18==0 ) { // case of pad 17 must shared to next trap chip as adc 20
                          //check trap is initialised.
                              adc=20-(pad % 18)-1;
                              LOG(debug1) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!....................................... setting data for simulator : " << trapindex+1 << " and adc : " << adc;
                              mTrapSimulator[trapindex+1].setData(adc,digititerator->getADC());
                          //    mTrapSimulator[trapindex].printAdcDatHuman(cout);
                       //       LOG(debug3) << "shared pad: " << pad << " pad : " << pad << " robfromshared:" << mfeeparam->getROBfromSharedPad(row,pad) << " mcmfromshared:" << mfeeparam->getMCMfromSharedPad(row,pad) << " adc : " << adc;
                          }

                     //  LOG(debug3) << "adding data for adc : " << adc << " for row,mcm combo of : "<< row <<"," << mcm;
                      }
                      olddetector=detector;
                      oldrob=rob;
                      oldmcm=mcm;
                      oldrow=row;
                      oldpad=pad;
              }
               
               LOG(info) << "and we are leaving the run method of TRDDPLTrapSimulatorTask \\o/ ";
           //    cout << ".... and we are in the run method ....." << endl;

           }
       private:
           std::array<TrapSimulator,8> mTrapSimulator;
           FeeParam *mfeeparam;
};



o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(int channelfan) 
{
    return DataProcessorSpec{"TRAP",Inputs{ InputSpec{"digitinput","TRD","DIGITS",0},
                                            InputSpec{"labelinput","TRD","LABELS",0},
                                            InputSpec{"labelinput","TRD","TRGLABELS",0}
                                          },
                                    Outputs{OutputSpec{"TRD","TRACKLETS",0}},
                                          AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()}};
}

} //end namespace trd
} //end namespace o2
