/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * ConditionsMQServer.cxx
 *
 * @since 2016-01-11
 * @author R. Grosso, from parmq/ParameterMQServer.cxx
 */

#include "TMessage.h"
#include "Rtypes.h"

#include "CCDB/ConditionsMQServer.h"
#include "FairMQLogger.h"

#include "CCDB/IdPath.h"
#include "CCDB/Condition.h"

using namespace AliceO2::CDB;
using std::endl;
using std::cout;
using std::string;

ConditionsMQServer::ConditionsMQServer() : ParameterMQServer(), fCdbManager(AliceO2::CDB::Manager::Instance()) {}

void ConditionsMQServer::InitTask()
{
  ParameterMQServer::InitTask();
  // Set first input
  if (GetProperty(FirstInputType, "") == "OCDB") {
    fCdbManager->setDefaultStorage(GetProperty(FirstInputName, "").c_str());
  }

  // Set second input
  if (GetProperty(SecondInputType, "") == "OCDB") {
    fCdbManager->setDefaultStorage(GetProperty(SecondInputName, "").c_str());
  }

  // Set output
  if (GetProperty(OutputName, "") != "") {
    if (GetProperty(OutputType, "") == "OCDB") {
      fCdbManager->setDefaultStorage(GetProperty(OutputName, "").c_str());
    }
  }
}

void free_tmessage(void* data, void* hint) { delete static_cast<TMessage*>(hint); }

void ConditionsMQServer::Run()
{
  std::string parameterName = "";
  Condition* aCondition = nullptr;

  while (CheckCurrentState(RUNNING)) {
    std::unique_ptr<FairMQMessage> req(fTransportFactory->CreateMessage());

    if (fChannels.at("data").at(0).Receive(req) > 0) {
      std::string reqStr(static_cast<char*>(req->GetData()), req->GetSize());
      LOG(INFO) << "Received parameter request from client: \"" << reqStr << "\"";

      size_t pos = reqStr.rfind(",");
      string newParameterName = reqStr.substr(0, pos);
      int runId = std::stoi(reqStr.substr(pos + 1));
      LOG(INFO) << "Parameter name: " << newParameterName;
      LOG(INFO) << "Run ID: " << runId;

      LOG(INFO) << "Retrieving parameter...";
      fCdbManager->setRun(runId);
      aCondition = fCdbManager->getObject(IdPath(newParameterName), runId);

      if (aCondition) {
        LOG(INFO) << "Sending following parameter to the client:";
        aCondition->printConditionMetaData();
        TMessage* tmsg = new TMessage(kMESS_OBJECT);
        tmsg->WriteObject(aCondition);

        std::unique_ptr<FairMQMessage> reply(
          fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

        fChannels.at("data").at(0).Send(reply);
      }
      else {
        LOG(ERROR) << "Could not get a condition for \"" << parameterName << "\" and run " << runId << "!";
      }
    }
  }
}

ConditionsMQServer::~ConditionsMQServer() { delete fCdbManager; }
