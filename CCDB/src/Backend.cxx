// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Backend.cxx
/// \brief Implementation of the Backend class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#include "CCDB/Backend.h"
#include "request.pb.h"

using namespace o2::ccdb;
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
