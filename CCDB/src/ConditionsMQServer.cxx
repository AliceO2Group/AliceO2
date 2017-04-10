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
 * @author R. Grosso, C. Kouzinopoulos from parmq/ParameterMQServer.cxx
 */

#include "TMessage.h"
#include "Rtypes.h"

#include "CCDB/Condition.h"
#include "CCDB/ConditionsMQServer.h"
#include "CCDB/IdPath.h"
#include <FairMQLogger.h>
#include <FairMQPoller.h>

// Google protocol buffers headers
#include <google/protobuf/stubs/common.h>
#include "request.pb.h"

#include <boost/algorithm/string.hpp>

using namespace o2::CDB;
using std::endl;
using std::cout;
using std::string;

ConditionsMQServer::ConditionsMQServer() : ParameterMQServer(), mCdbManager(o2::CDB::Manager::Instance()) {}

void ConditionsMQServer::InitTask()
{
  ParameterMQServer::InitTask();
  // Set first input
  if (GetProperty(FirstInputType, "") == "OCDB") {
    mCdbManager->setDefaultStorage(GetProperty(FirstInputName, "").c_str());
  }

  // Set second input
  if (GetProperty(SecondInputType, "") == "OCDB") {
    mCdbManager->setDefaultStorage(GetProperty(SecondInputName, "").c_str());
  }

  // Set output
  if (GetProperty(OutputName, "") != "") {
    if (GetProperty(OutputType, "") == "OCDB") {
      mCdbManager->setDefaultStorage(GetProperty(OutputName, "").c_str());
    }
  }
}

void free_tmessage(void* data, void* hint) { delete static_cast<TMessage*>(hint); }

void ConditionsMQServer::ParseDataSource(std::string& dataSource, const std::string& data)
{
  messaging::RequestMessage* msgReply = new messaging::RequestMessage;
  msgReply->ParseFromString(data);

  LOG(DEBUG) << "Data source: " << msgReply->datasource();

  dataSource = msgReply->datasource();

  delete msgReply;
}

void ConditionsMQServer::Deserialize(const std::string& messageString, std::string& object)
{
  messaging::RequestMessage* requestMessage = new messaging::RequestMessage;
  requestMessage->ParseFromString(messageString);

  object.assign(requestMessage->key());

  delete requestMessage;
}

void ConditionsMQServer::Run()
{
  std::unique_ptr<FairMQPoller> poller(
    fTransportFactory->CreatePoller(fChannels, { "data-put", "data-get", "broker-get" }));

  while (CheckCurrentState(RUNNING)) {

    poller->Poll(-1);

    if (poller->CheckInput("data-get", 0)) {
      std::unique_ptr<FairMQMessage> input(fTransportFactory->CreateMessage());

      if (Receive(input, "data-get") > 0) {
        std::string serialString(static_cast<char*>(input->GetData()), input->GetSize());

        //LOG(DEBUG) << "Received a GET client message: " << serialString;

        std::string dataSource;
        ParseDataSource(dataSource, serialString);

        if (dataSource == "OCDB") {
          // Retrieve the key from the serialized message
          std::string key;
          Deserialize(serialString, key);

          getFromOCDB(key);
        } else if (dataSource == "Riak") {
          // No need to de-serialize, just forward message to the broker
          fChannels.at("broker-get").at(0).Send(input);
        }
      }
    }

    if (poller->CheckInput("data-put", 0)) {
      std::unique_ptr<FairMQMessage> input(fTransportFactory->CreateMessage());

      if (Receive(input, "data-put") > 0) {
        std::string serialString(static_cast<char*>(input->GetData()), input->GetSize());

        // LOG(DEBUG) << "Received a PUT client message: " << serialString;
        LOG(DEBUG) << "Message size: " << input->GetSize();

        std::string dataSource;
        ParseDataSource(dataSource, serialString);

        if (dataSource == "OCDB") {
          LOG(ERROR) << "The GET operation is not supported for the OCDB data source yet";
        } else if (dataSource == "Riak") {
          fChannels.at("broker-put").at(0).Send(input);
        }
      }
    }

    if (poller->CheckInput("broker-get", 0)) {
      std::unique_ptr<FairMQMessage> input(fTransportFactory->CreateMessage());

      if (Receive(input, "broker-get") > 0) {
        LOG(DEBUG) << "Received object from broker with a size of: " << input->GetSize();

        fChannels.at("data-get").at(0).Send(input);
      }
    }
  }
}

// Query OCDB for the condition
void ConditionsMQServer::getFromOCDB(std::string key)
{
  // Change key from i.e. "/DET/Calib/Histo/Run2008_2008_v1_s0" to (DET/Calib/Histo, 2008)
  // FIXME: This will have to be changed in the future by adapting IdPath and getObject accordingly
  std::size_t pos = key.rfind("/");
  std::string identifier = key.substr(0, pos);
  key.erase(0, pos + 4);
  std::size_t pos2 = key.find("_");
  int runId = atoi(key.substr(0, pos2).c_str());

  Condition* aCondition = nullptr;

  mCdbManager->setRun(runId);
  aCondition = mCdbManager->getObject(IdPath(identifier), runId);

  if (aCondition) {
    LOG(DEBUG) << "Sending following parameter to the client:";
    aCondition->printConditionMetaData();
    TMessage* tmsg = new TMessage(kMESS_OBJECT);
    tmsg->WriteObject(aCondition);

    std::unique_ptr<FairMQMessage> message(
      fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

    fChannels.at("data-get").at(0).Send(message);
  } else {
    LOG(ERROR) << R"(Could not get a condition for ")" << key << R"(" and run )" << runId << "!";
  }
}

ConditionsMQServer::~ConditionsMQServer() { delete mCdbManager; }
