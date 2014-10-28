/**
 * FrameBuilder.cxx
 *
 * @since 2014-10-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FairMQPoller.h"

#include "FrameBuilder.h"

using namespace AliceO2::Devices;

FrameBuilder::FrameBuilder()
{
}

void FrameBuilder::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);

  int received = 0;
  int noOfMsgParts = fNumInputs - 1;

  while (fState == RUNNING) {
    FairMQMessage* msg = fTransportFactory->CreateMessage();

    poller->Poll(100);

    for (int i = 0; i < fNumInputs; ++i) {
      if (poller->CheckInput(i)) {
        received = fPayloadInputs->at(i)->Receive(msg);
        if (received > 0) {
          if (i < noOfMsgParts) {
            fPayloadOutputs->at(0)->Send(msg, "snd-more");
          } else {
            fPayloadOutputs->at(0)->Send(msg);
          }
        }
      }
    }

    delete msg;
  }

  delete poller;

  rateLogger.interrupt();
  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

FrameBuilder::~FrameBuilder()
{
}
