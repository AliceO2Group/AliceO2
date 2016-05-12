/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * runConditionsServer.cxx
 *
 * @since 2015-12-10
 * @author D. Klein, A. Rybalchenko, R. Grosso, C. Kouzinopoulos
 */

#include <iostream>

#include "CCDB/ConditionsMQServer.h"
#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"

#ifdef NANOMSG
#include "FairMQTransportFactoryNN.h"
#else
#include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;
using namespace boost::program_options;
using namespace AliceO2::CDB;

int main(int argc, char** argv)
{
  FairMQProgOptions config;
  ConditionsMQServer server;

  try {
    std::string firstInputName;
    std::string firstInputType;
    std::string secondInputName;
    std::string secondInputType;
    std::string outputName;
    std::string outputType;
    std::string channelName;

    options_description serverOptions("Conditions MQ Server options");
    serverOptions.add_options()(
      "first-input-name", value<std::string>(&firstInputName)->default_value("first_input.root"),
      "First input file name")("first-input-type", value<std::string>(&firstInputType)->default_value("ROOT"),
                               "First input file type (ROOT/ASCII)")(
      "second-input-name", value<std::string>(&secondInputName)->default_value(""), "Second input file name")(
      "second-input-type", value<std::string>(&secondInputType)->default_value("ROOT"),
      "Second input file type (ROOT/ASCII)")("output-name", value<std::string>(&outputName)->default_value(""),
                                             "Output file name")(
      "output-type", value<std::string>(&outputType)->default_value("ROOT"), "Output file type")(
      "channel-name", value<std::string>(&channelName)->default_value("ROOT"), "Output channel name");

    config.AddToCmdLineOptions(serverOptions);
    config.ParseAll(argc, argv);

    server.fChannels = config.GetFairMQMap();
    LOG(INFO) << "Communication channels: " << server.fChannels.size();

    {
#ifdef NANOMSG
      FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
      FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif
      server.SetTransport(transportFactory);

      server.SetProperty(ConditionsMQServer::FirstInputName, firstInputName);
      server.SetProperty(ConditionsMQServer::FirstInputType, firstInputType);
      server.SetProperty(ConditionsMQServer::SecondInputName, secondInputName);
      server.SetProperty(ConditionsMQServer::SecondInputType, secondInputType);
      server.SetProperty(ConditionsMQServer::OutputName, outputName);
      server.SetProperty(ConditionsMQServer::OutputType, outputType);
      server.SetProperty(ConditionsMQServer::ChannelName, channelName);

      server.CatchSignals();
      server.SetConfig(config);

      server.ChangeState("INIT_DEVICE");
      server.WaitForEndOfState("INIT_DEVICE");

      server.ChangeState("INIT_TASK");
      server.WaitForEndOfState("INIT_TASK");

      server.ChangeState("RUN");
      server.InteractiveStateLoop();
    }
  }
  catch (std::exception& e) {
    LOG(ERROR) << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit";
    return 1;
  }

  return 0;
}
