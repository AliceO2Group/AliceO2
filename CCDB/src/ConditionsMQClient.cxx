/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * ConditionsMQClient.cpp
 *
 * @since 2016-01-11
 * @author R. Grosso, C. Kouzinopoulos from examples/MQ/7-parameters/FairMQExample7Client.cxx
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "CCDB/Condition.h"
#include "CCDB/ConditionsMQClient.h"
#include "CCDB/ObjectHandler.h"
#include "FairMQLogger.h"

// Google protocol buffers headers
#include <google/protobuf/stubs/common.h>
#include "request.pb.h"

#include "boost/filesystem.hpp"

#include "TBufferFile.h"
#include "TFile.h"
#include "TMessage.h"
#include "Rtypes.h"

using namespace AliceO2::CDB;
using namespace std;

ConditionsMQClient::ConditionsMQClient() : fRunId(0), fParameterName() {}

ConditionsMQClient::~ConditionsMQClient() {}

void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }

// special class to expose protected TMessage constructor
class WrapTMessage : public TMessage {
public:
  WrapTMessage(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
};

void ConditionsMQClient::Serialize(std::string*& messageString, const std::string& key,
                                   const std::string& object /*= std::vector<char>()*/)
{
  messaging::RequestMessage* requestMessage = new messaging::RequestMessage;
  requestMessage->set_command(fOperationType);
  requestMessage->set_datasource(fDataSource);
  requestMessage->set_key(key);

  if (object.length() > 0) {
    requestMessage->set_value(object);
  }

  requestMessage->SerializeToString(messageString);

  delete requestMessage;
}

void ConditionsMQClient::Deserialize(const std::string& messageString, std::string& object)
{
  messaging::RequestMessage* requestMessage = new messaging::RequestMessage;
  requestMessage->ParseFromString(messageString);

  object.assign(requestMessage->value());

  delete requestMessage;
}

void ConditionsMQClient::RunOCDB()
{
  static int runId = 2000;

  if (fOperationType == "GET") {
    std::string* messageString = new string();
    Serialize(messageString, fParameterName + "," + to_string(runId));

    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(
      const_cast<char*>(messageString->c_str()), messageString->length(), CustomCleanup, messageString));
    unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage());

    if (fChannels.at("data-get").at(0).Send(request) > 0) {
      if (fChannels.at("data-get").at(0).Receive(reply) > 0) {
        WrapTMessage tmsg(reply->GetData(), reply->GetSize());
        Condition* aCondition = (Condition*)(tmsg.ReadObject(tmsg.GetClass()));
        LOG(DEBUG) << "Received a condition from the server:";
        aCondition->printConditionMetaData();
      }
    }
  } else if (fOperationType == "PUT") {
    LOG(ERROR) << "The PUT operation is not supported for the OCDB backend yet";
    return;
  }

  runId++;
  if (runId == 2101) {
    runId = 2001;
  }
}

void ConditionsMQClient::RunRiak()
{
  ObjectHandler objHandler;

  boost::filesystem::path dataPath(fObjectPath);
  boost::filesystem::recursive_directory_iterator endIterator;

  // Traverse the filesystem and retrieve the name of each root found
  if (boost::filesystem::exists(dataPath) && boost::filesystem::is_directory(dataPath)) {
    for (static boost::filesystem::recursive_directory_iterator directoryIterator(dataPath);
         directoryIterator != endIterator; ++directoryIterator) {
      if (boost::filesystem::is_regular_file(directoryIterator->status())) {

        // Retrieve the key from the filename by erasing the directory structure and trimming the file extension
        std::string str = directoryIterator->path().string();
        str.erase(0, fObjectPath.length());
        std::size_t pos = str.rfind(".");
        std::string key = str.substr(0, pos);

        if (fOperationType == "GET") {
          std::string* messageString = new string();
          Serialize(messageString, key);

          unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(
            const_cast<char*>(messageString->c_str()), messageString->length(), CustomCleanup, messageString));
          unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage());

          if (fChannels.at("data-get").at(0).Send(request) > 0) {
            if (fChannels.at("data-get").at(0).Receive(reply) > 0) {
              LOG(DEBUG) << "Received a condition with a size of " << reply->GetSize();

              std::string brokerString(static_cast<char*>(reply->GetData()), reply->GetSize());

              // Deserialize the received string
              std::string compressedObject;
              Deserialize(brokerString, compressedObject);

              // Decompress the compressed object
              std::string object;
              objHandler.Decompress(object, compressedObject);
            }
          }
        } else if (fOperationType == "PUT") {
          // Load the AliCDBEntry object from disk
          std::string object;
          objHandler.GetObject(directoryIterator->path().string(), object);

          // Compress the object before storing to Riak
          std::string compressed_object;
          objHandler.Compress(object, compressed_object);

          std::string* messageString = new string();
          Serialize(messageString, key, compressed_object);

          unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(
            const_cast<char*>(messageString->c_str()), messageString->length(), CustomCleanup, messageString));

          if (fChannels.at("data-put").at(0).Send(request) > 0) {
            LOG(DEBUG) << "Message sent" << endl;
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "Path " << fObjectPath << " not existing or not a directory";
  }
}

void ConditionsMQClient::Run()
{
  while (CheckCurrentState(RUNNING)) {

    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();

    if (fDataSource == "OCDB") {
      RunOCDB();
    } else if (fDataSource == "Riak") {
      RunRiak();
    } else {
      LOG(ERROR) << "\"" << fDataSource << "\" is not a valid Data Source";
      return;
    }

    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    LOG(DEBUG) << " Time elapsed: " << (endTime - startTime).total_milliseconds() << "ms";
  }
}

void ConditionsMQClient::SetProperty(const int key, const string& value)
{
  switch (key) {
    case ParameterName:
      fParameterName = value;
      break;
    case OperationType:
      fOperationType = value;
      break;
    case DataSource:
      fDataSource = value;
      break;
    case ObjectPath:
      fObjectPath = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string ConditionsMQClient::GetProperty(const int key, const string& default_ /*= ""*/)
{
  switch (key) {
    case ParameterName:
      return fParameterName;
      break;
    case OperationType:
      return fOperationType;
      break;
    case DataSource:
      return fDataSource;
      break;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void ConditionsMQClient::SetProperty(const int key, const int value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int ConditionsMQClient::GetProperty(const int key, const int default_ /*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}
