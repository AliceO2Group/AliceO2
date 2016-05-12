/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * ConditionsMQClient.h
 *
 * @since 2016-01-11
 * @author R. Grosso, C. Kouzinopoulos from examples/MQ/7-parameters/FairMQExample7Client.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQCLIENT_H_
#define ALICEO2_CDB_CONDITIONSMQCLIENT_H_

#include <string>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace CDB {

class ConditionsMQClient : public FairMQDevice {
public:
  enum { ParameterName = FairMQDevice::Last, OperationType, DataSource, ObjectPath, Last };

  ConditionsMQClient();

  virtual ~ConditionsMQClient();

  virtual void SetProperty(const int key, const std::string& value);
  virtual std::string GetProperty(const int key, const std::string& default_ = "");
  virtual void SetProperty(const int key, const int value);
  virtual int GetProperty(const int key, const int default_ = 0);

protected:
  virtual void Run();

private:
  int fRunId;
  std::string fParameterName;
  std::string fOperationType;
  std::string fDataSource;
  std::string fObjectPath;

  /// Serializes a key (and optionally value) to an std::string using Protocol Buffers
  void Serialize(std::string*& messageString, const std::string& key, const std::string& object = std::string());

  /// Deserializes a message and stores the value to an std::string using Protocol Buffers
  void Deserialize(const std::string& messageString, std::string& object);

  /// Run loop when an OCDB backend is chosen
  void RunOCDB();

  /// Run loop when a Riak backend is chosen
  void RunRiak();
};
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQCLIENT_H_ */
