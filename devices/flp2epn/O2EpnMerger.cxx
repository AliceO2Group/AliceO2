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
  std::unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels.at("data-in")));

  int numParts = fChannels.at("data-in").size() - 1;

  while (CheckCurrentState(RUNNING)) {
    std::unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());

    poller->Poll(100);

    for (int i = 0; i < fChannels.at("data-in").size(); i++) {
      if (poller->CheckInput(i)) {
        if (fChannels.at("data-in").at(i).Receive(msg)) {
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

O2EpnMerger::~O2EpnMerger()
{
}
