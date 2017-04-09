/**
 * ConditionsMQClient.h
 *
 * @since 2016-01-11
 * @author R. Grosso, C. Kouzinopoulos from examples/MQ/7-parameters/FairMQExample7Client.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQCLIENT_H_
#define ALICEO2_CDB_CONDITIONSMQCLIENT_H_

#include <string>

#include <FairMQDevice.h>

namespace o2 {
namespace CDB {

class ConditionsMQClient : public FairMQDevice {
public:
  ConditionsMQClient();
  virtual ~ConditionsMQClient();

protected:
  virtual void InitTask();
  virtual void Run();

private:
  int mRunId;
  std::string mParameterName;
  std::string mOperationType;
  std::string mDataSource;
  std::string mObjectPath;

};
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQCLIENT_H_ */
