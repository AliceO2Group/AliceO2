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
  FairMQChannel& inChannel = fChannels.at("data-in").at(0);
  FairMQChannel& outChannel = fChannels.at("data-out").at(0);

  while (CheckCurrentState(RUNNING)) {
    // int i = 0;
    bool more = false;

    do {
      /* Create an empty message to hold the message part */
      std::unique_ptr<FairMQMessage> part(fTransportFactory->CreateMessage());
      /* Block until a message is available to be received from socket */
      inChannel.Receive(part);
      /* Determine if more message parts are to follow */
      more = inChannel.ExpectsAnotherPart();
      // LOG(INFO) << "------ Get Msg Part "<< " more = " << more << " counter " << i++ ;
      if (more) {
          outChannel.SendPart(part);
      } else {
          outChannel.Send(part);
      }
    } while (more);
    // i = 0;
  }
}
