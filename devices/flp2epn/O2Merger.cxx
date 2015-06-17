/**
 * O2Merger.cxx
 *
 * @since 2012-12-06
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "O2Merger.h"
#include "FairMQPoller.h"
O2Merger::O2Merger()
{
}

O2Merger::~O2Merger()
{
}

void O2Merger::Run()
{
  FairMQPoller* poller = fTransportFactory->CreatePoller(fChannels["data-in"]);

  int NoOfMsgParts = fChannels["data-in"].size() - 1;

  while (GetCurrentState() == RUNNING) {
    FairMQMessage* msg = fTransportFactory->CreateMessage();

    poller->Poll(100);

    for (int i = 0; i < fChannels["data-in"].size(); i++) {
      if (poller->CheckInput(i)) {
        if (fChannels["data-in"].at(i).Receive(msg)) {
          // LOG(INFO) << "------ recieve Msg from " << i ;
          if (i < NoOfMsgParts) {
            fChannels["data-out"].at(0).Send(msg, "snd-more");
            //    LOG(INFO) << "------ Send  Msg Part " << i ;
          } else {
            fChannels["data-out"].at(0).Send(msg);
            //    LOG(INFO) << "------ Send  last Msg Part " << i ;
          }
        }
      }
    }

    delete msg;
  }

  delete poller;
}

