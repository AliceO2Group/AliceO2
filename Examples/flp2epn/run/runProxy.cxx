/**
 * runProxy.cxx
 *
 * @since 2013-10-07
 * @author A. Rybalchenko
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQProgOptions.h"
#include "FairMQProxy.h"
#include "runSimpleMQStateMachine.h"

using namespace boost::program_options;

int main(int argc, char** argv)
{
  try {
    int multipart;

    options_description proxyOptions("Proxy options");
    proxyOptions.add_options()
      ("multipart", value<int>(&multipart)->default_value(1), "Handle multipart payloads");

    FairMQProgOptions config;
    config.AddToCmdLineOptions(proxyOptions);
    config.ParseAll(argc, argv);

    FairMQProxy proxy;
    proxy.SetProperty(FairMQProxy::Multipart, multipart);
    runStateMachine(proxy, config);
  } catch (std::exception& e) {
    LOG(ERROR) << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit";
    return 1;
  }

  return 0;
}
