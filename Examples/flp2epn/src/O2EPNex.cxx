/**
 * O2EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <boost/thread.hpp>

#include "FairMQLogger.h"
#include "flp2epn/O2EPNex.h"
#include "O2FLPExContent.h"
using namespace std;

O2EPNex::O2EPNex()
{
}

void O2EPNex::Run()
{
  while (CheckCurrentState(RUNNING)) {
    unique_ptr<FairMQMessage> msg(NewMessage());

    fChannels.at("data").at(0).Receive(msg);

    // int numInput = msg->GetSize() / sizeof(O2FLPExContent);
    // O2FLPExContent* input = static_cast<O2FLPExContent*>(msg->GetData());

    // for (int i = 0; i < numInput; ++i) {
    //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
    // }
  }
}

O2EPNex::~O2EPNex()
{
}
