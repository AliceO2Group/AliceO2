/**
 * O2EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <boost/thread.hpp>

#include "FairMQLogger.h"
#include "flp2epn/O2EPNex.h"

struct Content
{
    double a;
    double b;
    int x;
    int y;
    int z;
};

O2EPNex::O2EPNex()
{
}

void O2EPNex::Run()
{
  while (CheckCurrentState(RUNNING)) {
    std::unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());

    fChannels.at("data-in").at(0).Receive(msg);

    // int numInput = msg->GetSize() / sizeof(Content);
    // Content* input = static_cast<Content*>(msg->GetData());

    // for (int i = 0; i < numInput; ++i) {
    //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
    // }
  }
}

O2EPNex::~O2EPNex()
{
}
