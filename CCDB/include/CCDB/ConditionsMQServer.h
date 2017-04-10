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

namespace o2 {
namespace CDB {

class ConditionsMQServer : public ParameterMQServer {
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
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQSERVER_H_ */
