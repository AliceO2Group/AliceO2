/**
 * runFLP.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"
#include "O2FLPex.h"

using namespace boost::program_options;

int main(int argc, char** argv)
{
  O2FLPex flp;
  flp.CatchSignals();

  FairMQProgOptions config;

  try {
    int numContent;

    options_description flpOptions("FLPex options");
    flpOptions.add_options()
      ("num-content", value<int>(&numContent)->default_value(1000), "Event size in bytes");

    config.AddToCmdLineOptions(flpOptions);

    config.ParseAll(argc, argv);
    flp.SetConfig(config);
    flp.SetProperty(O2FLPex::NumContent, numContent);

    flp.ChangeState("INIT_DEVICE");
    flp.WaitForEndOfState("INIT_DEVICE");

    flp.ChangeState("INIT_TASK");
    flp.WaitForEndOfState("INIT_TASK");

    flp.ChangeState("RUN");
    flp.InteractiveStateLoop();
  } catch (std::exception& e) {
    LOG(ERROR) << e.what();
    LOG(INFO) << "Command line options are the following: ";
    config.PrintHelp();
     return 1;
  }

  return 0;
}