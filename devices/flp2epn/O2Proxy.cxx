/**
 * O2Proxy.cxx
 *
 * @since 2013-10-02
 * @author A. Rybalchenko, M.Al-Turany
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "O2Proxy.h"



O2Proxy::O2Proxy()
{
}

O2Proxy::~O2Proxy()
{
}

void O2Proxy::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  while ( fState == RUNNING ) {
    // int i = 0;
    int64_t more = 0;
    size_t more_size = sizeof more;
    do {
      /* Create an empty ØMQ message to hold the message part */
      FairMQMessage* msgpart = fTransportFactory->CreateMessage();
      /* Block until a message is available to be received from socket */
      fPayloadInputs->at(0)->Receive(msgpart);
      /* Determine if more message parts are to follow */
      fPayloadInputs->at(0)->GetOption("rcv-more", &more, &more_size);
      // LOG(INFO) << "------ Get Msg Part "<< " more = " << more << " counter " << i++ ;
      if(more){
          fPayloadOutputs->at(0)->Send(msgpart, "snd-more");
      }else{
          fPayloadOutputs->at(0)->Send(msgpart);
      }
      delete msgpart;
    } while (more);
    // i = 0;
  }

  rateLogger.interrupt();
  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}
