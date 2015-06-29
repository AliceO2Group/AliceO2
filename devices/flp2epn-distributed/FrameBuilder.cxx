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
  FairMQPoller* poller = fTransportFactory->CreatePoller(fChannels.at("data-in"));

  fNumInputs = fChannels.at("data-in").size();
  int noOfMsgParts = fNumInputs - 1;

  while (CheckCurrentState(RUNNING)) {
    FairMQMessage* msg = fTransportFactory->CreateMessage();

    poller->Poll(100);

    for (int i = 0; i < fNumInputs; ++i) {
      if (poller->CheckInput(i)) {
        if (fChannels.at("data-in").at(i).Receive(msg) > 0) {
          if (i < noOfMsgParts) {
            fChannels.at("data-out").at(0).Send(msg, "snd-more");
          } else {
            fChannels.at("data-out").at(0).Send(msg);
          }
        }
      }
    }

    delete msg;
  }

  delete poller;
}

FrameBuilder::~FrameBuilder()
{
}
