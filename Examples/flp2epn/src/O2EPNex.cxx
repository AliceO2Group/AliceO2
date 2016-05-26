/**
 * O2EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <boost/thread.hpp>

#include "FairMQLogger.h"
#include "O2EPNex.h"
#include "O2FLPExContent.h"

<<<<<<< HEAD:Examples/flp2epn/O2EPNex.cxx
using namespace std;
=======
struct Content
{
    double a;
    double b;
    int x;
    int y;
    int z;
};
>>>>>>> 685fc34... Apply new scheme to flp2epn:Examples/flp2epn/src/O2EPNex.cxx

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
