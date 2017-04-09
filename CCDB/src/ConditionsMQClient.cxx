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
#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>

#include "boost/filesystem.hpp"

using namespace o2::CDB;
using namespace std;

ConditionsMQClient::ConditionsMQClient() : mRunId(0), mParameterName() {}

ConditionsMQClient::~ConditionsMQClient() = default;

void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }

void ConditionsMQClient::InitTask()
{
  mParameterName = GetConfig()->GetValue<string>("parameter-name");
  mOperationType = GetConfig()->GetValue<string>("operation-type");
  mDataSource = GetConfig()->GetValue<string>("data-source");
  mObjectPath = GetConfig()->GetValue<string>("object-path");
}

void ConditionsMQClient::Run()
{
  Backend* backend;

  while (CheckCurrentState(RUNNING)) {

    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();

    if (mDataSource == "OCDB") {
      backend = new BackendOCDB();
    } else if (mDataSource == "Riak") {
      backend = new BackendRiak();
    } else {
      LOG(ERROR) << R"(")" << mDataSource << R"(" is not a valid Data Source)";
      return;
    }

    boost::filesystem::path dataPath(mObjectPath);
    boost::filesystem::recursive_directory_iterator endIterator;

    // Traverse the filesystem and retrieve the name of each root found
    if (boost::filesystem::exists(dataPath) && boost::filesystem::is_directory(dataPath)) {
      for (static boost::filesystem::recursive_directory_iterator directoryIterator(dataPath);
           directoryIterator != endIterator; ++directoryIterator) {
        if (boost::filesystem::is_regular_file(directoryIterator->status())) {

          // Retrieve the key from the filename by erasing the directory structure and trimming the file extension
          std::string str = directoryIterator->path().string();
          str.erase(0, mObjectPath.length());
          std::size_t pos = str.rfind(".");
          std::string key = str.substr(0, pos);

          if (mOperationType == "GET") {
            std::string* messageString = new string();
            backend->Serialize(messageString, key, mOperationType, mDataSource);

            unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(
              const_cast<char*>(messageString->c_str()), messageString->length(), CustomCleanup, messageString));
            unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage());

            if (fChannels.at("data-get").at(0).Send(request) > 0) {
              if (fChannels.at("data-get").at(0).Receive(reply) > 0) {
                LOG(DEBUG) << "Received a condition with a size of " << reply->GetSize();
                backend->UnPack(std::move(reply));
              }
            }
          } else if (mOperationType == "PUT") {
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
      LOG(ERROR) << "Path " << mObjectPath << " not existing or not a directory";
    }

    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    LOG(DEBUG) << " Time elapsed: " << (endTime - startTime).total_milliseconds() << "ms";
  }
}
