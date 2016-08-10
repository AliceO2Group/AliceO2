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
 * @author D. Klein, A. Rybalchenko, R. Grosso
 */

#include <iostream>

#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"
#include "ConditionsMQServer.h"
#include "TApplication.h"
using namespace std;
using namespace boost::program_options;
using namespace AliceO2::CDB;

int main(int argc, char** argv)
{
    ConditionsMQServer server;
    server.CatchSignals();

    FairMQProgOptions config;

    try
    {
        string firstInputName;
        string firstInputType;
        string secondInputName;
        string secondInputType;
        string outputName;
        string outputType;

        options_description serverOptions("Parameter MQ Server options");
        serverOptions.add_options()
            ("first-input-name", value<string>(&firstInputName)->default_value("first_input.root"), "First input file name")
            ("first-input-type", value<string>(&firstInputType)->default_value("ROOT"), "First input file type (ROOT/ASCII)")
            ("second-input-name", value<string>(&secondInputName)->default_value(""), "Second input file name")
            ("second-input-type", value<string>(&secondInputType)->default_value("ROOT"), "Second input file type (ROOT/ASCII)")
            ("output-name", value<string>(&outputName)->default_value(""), "Output file name")
            ("output-type", value<string>(&outputType)->default_value("ROOT"), "Output file type");

        config.AddToCmdLineOptions(serverOptions);
        config.ParseAll(argc, argv);

        string file = config.GetValue<string>("config-json-file");
        string id = config.GetValue<string>("id");

        config.UserParser<FairMQParser::JSON>(file, id);

        server.fChannels = config.GetFairMQMap();

        LOG(INFO) << "PID: " << getpid();

        TApplication app("ConditionsMQServer", 0, 0);

        server.SetProperty(ConditionsMQServer::Id, id);
        server.SetProperty(ConditionsMQServer::NumIoThreads, config.GetValue<int>("io-threads"));

        server.SetProperty(ConditionsMQServer::FirstInputName, firstInputName);
        server.SetProperty(ConditionsMQServer::FirstInputType, firstInputType);
        server.SetProperty(ConditionsMQServer::SecondInputName, secondInputName);
        server.SetProperty(ConditionsMQServer::SecondInputType, secondInputType);
        server.SetProperty(ConditionsMQServer::OutputName, outputName);
        server.SetProperty(ConditionsMQServer::OutputType, outputType);

        server.ChangeState("INIT_DEVICE");
        server.WaitForEndOfState("INIT_DEVICE");

        server.ChangeState("INIT_TASK");
        server.WaitForEndOfState("INIT_TASK");

        server.ChangeState("RUN");
        server.InteractiveStateLoop();

        gApplication->Terminate();
    }
    catch (exception& e)
    {
        LOG(ERROR) << e.what();
        LOG(INFO) << "Command line options are the following: ";
        config.PrintHelp();
        return 1;
    }

    return 0;
}
