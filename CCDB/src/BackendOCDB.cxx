// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BackendOCDB.cxx
/// \brief Implementation of the BackendOCDB class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#include "CCDB/BackendOCDB.h"
#include "CCDB/Condition.h"
#include "CCDB/ObjectHandler.h"

#include "TMessage.h"

#include <FairMQLogger.h>

#include <zlib.h>

using namespace o2::CDB;
using namespace std;

// special class to expose protected TMessage constructor
class WrapTMessage : public TMessage {
public:
  WrapTMessage(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
};

BackendOCDB::BackendOCDB() {}

void BackendOCDB::Pack(const std::string& path, const std::string& key, std::string*& messageString)
{
  LOG(ERROR) << "The PUT operation is not supported for the OCDB backend yet";
}

void BackendOCDB::UnPack(std::unique_ptr<FairMQMessage> msg)
{
  WrapTMessage tmsg(msg->GetData(), msg->GetSize());
  Condition* aCondition = (Condition*)(tmsg.ReadObject(tmsg.GetClass()));
  LOG(DEBUG) << "Received a condition from the server:";
  aCondition->printConditionMetaData();
}
