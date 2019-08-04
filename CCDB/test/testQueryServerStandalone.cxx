// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <CCDB/BackendOCDB.h>
#include <CCDB/Condition.h>
#include <FairMQChannel.h>
#include <FairMQLogger.h>
#include <FairMQParts.h>
#include <FairMQTransportFactory.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include "TestClass.h"

using namespace std;

void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }
void queryConditionServer(string transport, string address)
{
  auto factory = FairMQTransportFactory::CreateTransportFactory(transport);
  auto channel = FairMQChannel{"data-get", "req", factory};
  channel.Connect(address);
  channel.ValidateChannel();

  auto backend = new o2::ccdb::BackendOCDB();

  std::string* messageString = new string();
  std::string operationType("GET");
  // FIXME: how to setup the key?
  // just a test key -- corresponding to example from unit test testWriteReadAny.cxx
  std::string key("TestParam/Test/Test/Run1_1_v1_s0");
  std::string source("OCDB");
  backend->Serialize(messageString, key, operationType, source);

  std::cerr << messageString->c_str() << "\n";
  unique_ptr<FairMQMessage> request(factory->CreateMessage(const_cast<char*>(messageString->c_str()),
                                                           messageString->length(), CustomCleanup, messageString));

  unique_ptr<FairMQMessage> reply(factory->CreateMessage());

  channel.Send(request);
  if (channel.Receive(reply) > 0) {
    LOG(DEBUG) << "Received a condition with a size of " << reply->GetSize();
    auto condition = backend->UnPack(std::move(reply));
    LOG(DEBUG) << "TYPE " << condition->getObject()->IsA()->GetName() << "\n";

    // retrieve concrete type
    TestClass* c = nullptr;
    condition->getObjectAs(c);
    if (c) {
      LOG(DEBUG) << "RECEIVED parameter value is " << c->mD << "\n";
    }
  }
}

int main()
{
  // assuming that conditions-server is running and listening on port 25006
  // see conditions-server.json
  queryConditionServer("zeromq", "tcp://localhost:25006");
}
