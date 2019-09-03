// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * ConditionsMQServer.h
 *
 * @since 2016-01-11
 * @author R. Grosso, C. Kouzinopoulos from parmq/ParameterMQServer.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQSERVER_H_
#define ALICEO2_CDB_CONDITIONSMQSERVER_H_

#include <string>

#include "CCDB/Manager.h"
#include "ParameterMQServer.h"

namespace o2
{
namespace ccdb
{

class ConditionsMQServer : public ParameterMQServer
{
 public:
  ConditionsMQServer();

  ~ConditionsMQServer() override;

  void Run() override;

  void InitTask() override;

 private:
  Manager* mCdbManager;

  void getFromOCDB(std::string key);

  /// Parses a serialized message for a data source entry
  void ParseDataSource(std::string& dataSource, const std::string& data);

  /// Deserializes a message and stores the value to an std::string using Protocol Buffers
  void Deserialize(const std::string& messageString, std::string& key);
};
} // namespace ccdb
} // namespace o2

#endif /* ALICEO2_CDB_CONDITIONSMQSERVER_H_ */
