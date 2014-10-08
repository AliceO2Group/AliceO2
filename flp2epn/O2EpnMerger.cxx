/**
 * O2EpnMerger.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "O2EpnMerger.h"
#include "FairMQLogger.h"

O2EpnMerger::O2EpnMerger()
{
}

void O2EpnMerger::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
    //--------------------
    
  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);
    
  bool received = false;
  int NoOfMsgParts=fNumInputs-1;
    
  while ( fState == RUNNING ) {
    FairMQMessage* msg = fTransportFactory->CreateMessage();
    poller->Poll(100);
        for(int i = 0; i < fNumInputs; i++) {
            if (poller->CheckInput(i)){
                received = fPayloadInputs->at(i)->Receive(msg);
                // LOG(INFO) << "------ recieve Msg from " << i ;
            }
            if (received) {
                if(i<NoOfMsgParts){
                    fPayloadOutputs->at(0)->Send(msg, "snd-more");
                    //    LOG(INFO) << "------ Send  Msg Part " << i ;
                }else{
                    fPayloadOutputs->at(0)->Send(msg);
                    //    LOG(INFO) << "------ Send  last Msg Part " << i ;
                }
                received = false;
            }
        }
        
        delete msg;
    }
    
    delete poller;
    
    rateLogger.interrupt();
    rateLogger.join();
//--------------------
    
    
    
  while ( fState == RUNNING ) {
    FairMQMessage* msg = fTransportFactory->CreateMessage();

    fPayloadInputs->at(0)->Receive(msg);

    int inputSize = msg->GetSize();
    int numInput = inputSize / sizeof(Content);
    Content* input = reinterpret_cast<Content*>(msg->GetData());

    // for (int i = 0; i < numInput; ++i) {
    //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
    // }

    delete msg;
  }

  rateLogger.interrupt();
  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

O2EpnMerger::~O2EpnMerger()
{
}
