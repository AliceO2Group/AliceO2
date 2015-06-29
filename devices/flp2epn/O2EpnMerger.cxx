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
  FairMQPoller* poller = fTransportFactory->CreatePoller(fChannels["data-in"]);

  bool received = false;
  int NoOfMsgParts = fChannels["data-in"].size() - 1;

  while (CheckCurrentState(RUNNING)) {
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

//--------------------

  // while (CheckCurrentState(RUNNING)) {
  //   FairMQMessage* msg = fTransportFactory->CreateMessage();

  //   fChannels["data-in"].at(0).Receive(msg);

  //   int inputSize = msg->GetSize();
  //   int numInput = inputSize / sizeof(Content);
  //   Content* input = reinterpret_cast<Content*>(msg->GetData());

  //   // for (int i = 0; i < numInput; ++i) {
  //   //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
  //   // }

  //   delete msg;
  // }
}

O2EpnMerger::~O2EpnMerger()
{
}
