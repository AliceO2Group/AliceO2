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

#include "FairRuntimeDb.h"
#include "FairParAsciiFileIo.h"
#include "FairParRootFileIo.h"

#include "ConditionsMQServer.h"
#include "FairMQLogger.h"
#include "FairParGenericSet.h"

#include "IdPath.h"
#include "Condition.h"

using namespace AliceO2::CDB;
using std::endl;
using std::cout;
using std::string;

ConditionsMQServer::ConditionsMQServer()
  : fRtdb(FairRuntimeDb::instance()),
    fCdbManager(AliceO2::CDB::Manager::Instance()),
    fFirstInputName("first_input.root"),
    fFirstInputType("ROOT"),
    fSecondInputName(""),
    fSecondInputType("ROOT"),
    fOutputName(""),
    fOutputType("ROOT")
{
}

void ConditionsMQServer::InitTask()
{
  if (fRtdb != 0) {
    // Set first input
    if (fFirstInputType == "ROOT") {
      FairParRootFileIo* par1R = new FairParRootFileIo();
      par1R->open(fFirstInputName.data(), "UPDATE");
      fRtdb->setFirstInput(par1R);
    }
    else if (fFirstInputType == "ASCII") {
      FairParAsciiFileIo* par1A = new FairParAsciiFileIo();
      par1A->open(fFirstInputName.data(), "in");
      fRtdb->setFirstInput(par1A);
    }
    else if (fFirstInputType == "OCDB") {
      fCdbManager->setDefaultStorage(fFirstInputName.c_str());
    }

    // Set second input
    if (fSecondInputName != "") {
      if (fSecondInputType == "ROOT") {
        FairParRootFileIo* par2R = new FairParRootFileIo();
        par2R->open(fSecondInputName.data(), "UPDATE");
        fRtdb->setSecondInput(par2R);
      }
      else if (fSecondInputType == "ASCII") {
        FairParAsciiFileIo* par2A = new FairParAsciiFileIo();
        par2A->open(fSecondInputName.data(), "in");
        fRtdb->setSecondInput(par2A);
      }
      else if (fSecondInputType == "OCDB") {
        fCdbManager->setDefaultStorage(fSecondInputName.c_str());
      }
    }

    // Set output
    if (fOutputName != "") {
      if (fOutputType == "ROOT") {
        FairParRootFileIo* parOut = new FairParRootFileIo(kTRUE);
        parOut->open(fOutputName.data());
        fRtdb->setOutput(parOut);
      }
      else if (fOutputType == "OCDB") {
        fCdbManager->setDefaultStorage(fOutputName.c_str());
      }

      fRtdb->saveOutput();
    }
  }
}

void free_tmessage(void* data, void* hint) { delete static_cast<TMessage*>(hint); }

void ConditionsMQServer::Run()
{
  std::string parameterName = "";
  FairParGenericSet* par = nullptr;
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
      // Check if the parameter name has changed to avoid getting same container repeatedly
      if (fFirstInputType != "OCDB") {
        if (newParameterName != parameterName) {
          parameterName = newParameterName;
          par = static_cast<FairParGenericSet*>(fRtdb->getContainer(parameterName.c_str()));
        }
        fRtdb->initContainers(runId);
      }
      else {
        fCdbManager->setRun(runId);
        aCondition = fCdbManager->getObject(IdPath(newParameterName), runId);
      }

      LOG(INFO) << "Sending following parameter to the client:";
      if (par) {
        par->print();

        TMessage* tmsg = new TMessage(kMESS_OBJECT);
	tmsg->WriteObject(par);

        std::unique_ptr<FairMQMessage> reply(
          fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

        fChannels.at("data").at(0).Send(reply);
      }
      else if (aCondition) {
	aCondition->printConditionMetaData();
	TMessage* tmsg = new TMessage(kMESS_OBJECT);
	tmsg->WriteObject(aCondition);

        std::unique_ptr<FairMQMessage> reply(
          fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

        fChannels.at("data").at(0).Send(reply);
      }
      else {
        LOG(ERROR) << "Parameter uninitialized!";
      }
    }
  }
}

void ConditionsMQServer::SetProperty(const int key, const std::string& value)
{
  switch (key) {
    case FirstInputName:
      fFirstInputName = value;
      break;
    case FirstInputType:
      fFirstInputType = value;
      break;
    case SecondInputName:
      fSecondInputName = value;
      break;
    case SecondInputType:
      fSecondInputType = value;
      break;
    case OutputName:
      fOutputName = value;
      break;
    case OutputType:
      fOutputType = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string ConditionsMQServer::GetProperty(const int key, const string& default_ /*= ""*/)
{
  switch (key) {
    case FirstInputName:
      return fFirstInputName;
    case FirstInputType:
      return fFirstInputType;
    case SecondInputName:
      return fSecondInputName;
    case SecondInputType:
      return fSecondInputType;
    case OutputName:
      return fOutputName;
    case OutputType:
      return fOutputType;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void ConditionsMQServer::SetProperty(const int key, const int value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int ConditionsMQServer::GetProperty(const int key, const int default_ /*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

ConditionsMQServer::~ConditionsMQServer()
{
  if (fRtdb) {
    delete fRtdb;
  }
  if (fCdbManager) {
    delete fCdbManager;
  }
}
