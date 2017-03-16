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

#include "CCDB/BackendOCDB.h"
#include "CCDB/BackendRiak.h"
#include "CCDB/ConditionsMQClient.h"
#include "FairMQLogger.h"
#include "FairMQProgOptions.h"

#include "boost/filesystem.hpp"

using namespace AliceO2::CDB;
using namespace std;

ConditionsMQClient::ConditionsMQClient() : fRunId(0), fParameterName() {}

ConditionsMQClient::~ConditionsMQClient() {}

void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }

void ConditionsMQClient::InitTask()
{
  fParameterName = GetConfig()->GetValue<string>("parameter-name");
  fOperationType = GetConfig()->GetValue<string>("operation-type");
  fDataSource = GetConfig()->GetValue<string>("data-source");
  fObjectPath = GetConfig()->GetValue<string>("object-path");
}

void ConditionsMQClient::Run()
{
  Backend* backend;

  while (CheckCurrentState(RUNNING)) {

    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();

    if (fDataSource == "OCDB") {
      backend = new BackendOCDB();
    } else if (fDataSource == "Riak") {
      backend = new BackendRiak();
    } else {
      LOG(ERROR) << "\"" << fDataSource << "\" is not a valid Data Source";
      return;
    }

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
            backend->Serialize(messageString, key, fOperationType, fDataSource);

            unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(
              const_cast<char*>(messageString->c_str()), messageString->length(), CustomCleanup, messageString));
            unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage());

            if (fChannels.at("data-get").at(0).Send(request) > 0) {
              if (fChannels.at("data-get").at(0).Receive(reply) > 0) {
                LOG(DEBUG) << "Received a condition with a size of " << reply->GetSize();
                backend->UnPack(std::move(reply));
              }
            }
          } else if (fOperationType == "PUT") {
            std::string* messageString = new string();
            backend->Pack(directoryIterator->path().string(), key, messageString);

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

    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    LOG(DEBUG) << " Time elapsed: " << (endTime - startTime).total_milliseconds() << "ms";
  }
}
