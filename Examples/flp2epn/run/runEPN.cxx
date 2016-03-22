/**
 * runEPN.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <iostream>

#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"
#include "O2EPNex.h"

using namespace std;

int main(int argc, char** argv)
{
  O2EPNex epn;
  epn.CatchSignals();

  FairMQProgOptions config;

  try {
    config.ParseAll(argc, argv);
    epn.SetConfig(config);

    epn.ChangeState("INIT_DEVICE");
    epn.WaitForEndOfState("INIT_DEVICE");

    epn.ChangeState("INIT_TASK");
    epn.WaitForEndOfState("INIT_TASK");

    epn.ChangeState("RUN");
    epn.InteractiveStateLoop();
  } catch (std::exception& e) {
    LOG(ERROR) << e.what();
    LOG(INFO) << "Command line options are the following: ";
    config.PrintHelp();
     return 1;
  }

  return 0;
}
