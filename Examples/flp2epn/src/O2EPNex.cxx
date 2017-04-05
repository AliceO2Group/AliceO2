/**
 * O2EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <FairMQLogger.h>
#include "flp2epn/O2EPNex.h"
#include "O2FLPExContent.h"

O2EPNex::O2EPNex()
{
  OnData("data", &O2EPNex::Process);
}

bool O2EPNex::Process(FairMQMessagePtr& msg, int index)
{
  // int numInput = msg->GetSize() / sizeof(O2FLPExContent);
  // O2FLPExContent* input = static_cast<O2FLPExContent*>(msg->GetData());

  // for (int i = 0; i < numInput; ++i) {
  //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
  // }

  return true;
}

O2EPNex::~O2EPNex()
= default;
