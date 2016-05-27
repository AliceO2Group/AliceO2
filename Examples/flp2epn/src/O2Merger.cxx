/**
 * O2Merger.cxx
 *
 * @since 2012-12-06
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FairMQPoller.h"
#include "flp2epn/O2Merger.h"

O2Merger::O2Merger()
{
}

O2Merger::~O2Merger()
{
}

void O2Merger::Run()
{
  std::unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels.at("data-in")));

  int numParts = fChannels.at("data-in").size() - 1;

  while (CheckCurrentState(RUNNING)) {
    std::unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());

    poller->Poll(100);

    for (unsigned int i = 0; i < fChannels.at("data-in").size(); i++) {
      if (poller->CheckInput(i)) {
        if (fChannels.at("data-in").at(i).Receive(msg) >= 0) {
          // LOG(INFO) << "------ recieve Msg from " << i ;
          if (i < numParts) {
            fChannels.at("data-out").at(0).SendPart(msg);
            //    LOG(INFO) << "------ Send  Msg Part " << i ;
          } else {
            fChannels.at("data-out").at(0).Send(msg);
            //    LOG(INFO) << "------ Send  last Msg Part " << i ;
          }
        }
      }
    }
  }
}

