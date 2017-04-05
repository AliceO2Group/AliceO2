/**
 * O2FLPex.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <vector>
#include <cstdlib>     /* srand, rand */
#include <ctime>       /* time */

#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>
#include "flp2epn/O2FLPex.h"
#include "O2FLPExContent.h"

O2FLPex::O2FLPex() :
  fNumContent(10000)
{
}

O2FLPex::~O2FLPex()
= default;

void O2FLPex::InitTask()
{
  srand(time(nullptr));

  fNumContent = GetConfig()->GetValue<int>("num-content");
  LOG(INFO) << "Message size (num-content * sizeof(O2FLPExContent)): " << fNumContent * sizeof(O2FLPExContent) << " bytes.";
}

bool O2FLPex::ConditionalRun()
{
  std::vector<O2FLPExContent> payload(fNumContent);

  for (int i = 0; i < fNumContent; ++i) {
    payload.at(i).x = rand() % 100 + 1;
    payload.at(i).y = rand() % 100 + 1;
    payload.at(i).z = rand() % 100 + 1;
    payload.at(i).a = (rand() % 100 + 1) / (rand() % 100 + 1);
    payload.at(i).b = (rand() % 100 + 1) / (rand() % 100 + 1);
    // LOG(INFO) << (&payload[i])->x << " " << (&payload[i])->y << " " << (&payload[i])->z << " " << (&payload[i])->a << " " << (&payload[i])->b;
  }

  FairMQMessagePtr msg(NewMessage(fNumContent * sizeof(O2FLPExContent)));
  memcpy(msg->GetData(), payload.data(), fNumContent * sizeof(O2FLPExContent));

  fChannels.at("data").at(0).Send(msg);

  return true;
}
