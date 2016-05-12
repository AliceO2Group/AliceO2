/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * runConditionsClient.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, R. Grosso, C. Kouzinopoulos
 */

#include <iostream>

#include "CCDB/ConditionsMQClient.h"
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
  ConditionsMQClient client;
  FairMQProgOptions config;

  try {
    std::string parameterName;
    std::string operationType;
    std::string dataSource;
    std::string objectPath;

    options_description clientOptions("Conditions MQ Client options");
    clientOptions.add_options()("parameter-name", value<string>(&parameterName)->default_value("DET/Calib/Histo"),
                                "Parameter Name");
    clientOptions.add_options()("operation-type", value<string>(&operationType)->default_value("GET"),
                                "Operation Type");
    clientOptions.add_options()("data-source", value<string>(&dataSource)->default_value("OCDB"), "Data Source");
    clientOptions.add_options()("object-path", value<string>(&objectPath)->default_value("OCDB"), "Object Path");

    config.AddToCmdLineOptions(clientOptions);
    config.ParseAll(argc, argv);

    client.fChannels = config.GetFairMQMap();
    LOG(INFO) << "Communication channels: " << client.fChannels.size();

    {
#ifdef NANOMSG
      FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
      FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif
      client.SetTransport(transportFactory);

      client.SetProperty(ConditionsMQClient::ParameterName, parameterName);
      client.SetProperty(ConditionsMQClient::OperationType, operationType);
      client.SetProperty(ConditionsMQClient::DataSource, dataSource);
      client.SetProperty(ConditionsMQClient::ObjectPath, objectPath);

      client.ChangeState("INIT_DEVICE");
      client.WaitForEndOfState("INIT_DEVICE");

      client.ChangeState("INIT_TASK");
      client.WaitForEndOfState("INIT_TASK");

      client.ChangeState("RUN");
      client.InteractiveStateLoop();
    }
  }
  catch (std::exception& e) {
    LOG(ERROR) << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit";
    return 1;
  }

  return 0;
}
