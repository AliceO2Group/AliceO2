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

using namespace std;

O2FLPex::O2FLPex() :
  fEventSize(10000)
{
}

O2FLPex::~O2FLPex()
{
}

void O2FLPex::Init()
{
  FairMQDevice::Init();
}

void O2FLPex::Run()
{
  srand(time(NULL));

  LOG(DEBUG) << "Message size: " << fEventSize * sizeof(Content) << " bytes.";

  while (CheckCurrentState(RUNNING)) {
    Content* payload = new Content[fEventSize];

    for (int i = 0; i < fEventSize; ++i) {
      (&payload[i])->x = rand() % 100 + 1;
      (&payload[i])->y = rand() % 100 + 1;
      (&payload[i])->z = rand() % 100 + 1;
      (&payload[i])->a = (rand() % 100 + 1) / (rand() % 100 + 1);
      (&payload[i])->b = (rand() % 100 + 1) / (rand() % 100 + 1);
      // LOG(INFO) << (&payload[i])->x << " " << (&payload[i])->y << " " << (&payload[i])->z << " " << (&payload[i])->a << " " << (&payload[i])->b;
    }

    FairMQMessage* msg = fTransportFactory->CreateMessage(fEventSize * sizeof(Content));
    memcpy(msg->GetData(), payload, fEventSize * sizeof(Content));

    fChannels["data-out"].at(0).Send(msg);

    delete[] payload;
    delete msg;
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
  case EventSize:
    fEventSize = value;
    break;
  default:
    FairMQDevice::SetProperty(key, value);
    break;
  }
}

int O2FLPex::GetProperty(const int key, const int default_/*= 0*/)
{
  switch (key) {
  case EventSize:
    return fEventSize;
  default:
    return FairMQDevice::GetProperty(key, default_);
  }
}
