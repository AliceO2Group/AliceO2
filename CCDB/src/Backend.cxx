/// \file Backend.cxx
/// \brief Implementation of the Backend class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#include "CCDB/Backend.h"

using namespace o2::CDB;
using namespace std;

void Backend::Serialize(std::string*& messageString, const std::string& key, const std::string& operationType,
                        const std::string& dataSource, const std::string& object /*= std::vector<char>()*/)
{
  messaging::RequestMessage* requestMessage = new messaging::RequestMessage;
  requestMessage->set_command(operationType);
  requestMessage->set_datasource(dataSource);
  requestMessage->set_key(key);

  if (object.length() > 0) {
    requestMessage->set_value(object);
  }

  requestMessage->SerializeToString(messageString);

  delete requestMessage;
}
