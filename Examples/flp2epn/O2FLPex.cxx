/**
 * O2FLPex.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "O2FLPex.h"
#include "O2FLPExContent.h"

using namespace std;

O2FLPex::O2FLPex() :
  fNumContent(10000)
{
}

O2FLPex::~O2FLPex()
{
}

void O2FLPex::Run()
{
  srand(time(NULL));

  FairMQChannel& outChannel = fChannels.at("data").at(0);

  LOG(DEBUG) << "Message size: " << fNumContent * sizeof(O2FLPExContent) << " bytes.";

  while (CheckCurrentState(RUNNING)) {
    vector<O2FLPExContent> payload(fNumContent);

    for (int i = 0; i < fNumContent; ++i) {
      payload.at(i).x = rand() % 100 + 1;
      payload.at(i).y = rand() % 100 + 1;
      payload.at(i).z = rand() % 100 + 1;
      payload.at(i).a = (rand() % 100 + 1) / (rand() % 100 + 1);
      payload.at(i).b = (rand() % 100 + 1) / (rand() % 100 + 1);
      // LOG(INFO) << (&payload[i])->x << " " << (&payload[i])->y << " " << (&payload[i])->z << " " << (&payload[i])->a << " " << (&payload[i])->b;
    }

    unique_ptr<FairMQMessage> msg(NewMessage(fNumContent * sizeof(O2FLPExContent)));
    memcpy(msg->GetData(), payload.data(), fNumContent * sizeof(O2FLPExContent));

    outChannel.Send(msg);
  }
}

void O2FLPex::SetProperty(const int key, const string& value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string O2FLPex::GetProperty(const int key, const string& default_/*= ""*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void O2FLPex::SetProperty(const int key, const int value)
{
  switch (key) {
    case NumContent:
      fNumContent = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int O2FLPex::GetProperty(const int key, const int default_/*= 0*/)
{
  switch (key) {
    case NumContent:
      return fNumContent;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}
