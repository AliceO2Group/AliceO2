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
 * ConditionsMQClient.h
 *
 * @since 2016-01-11
 * @author R. Grosso, C. Kouzinopoulos from examples/MQ/7-parameters/FairMQExample7Client.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQCLIENT_H_
#define ALICEO2_CDB_CONDITIONSMQCLIENT_H_

#include <string>

#include <FairMQDevice.h>

namespace o2
{
namespace ccdb
{

class ConditionsMQClient : public FairMQDevice
{
 public:
  ConditionsMQClient();
  ~ConditionsMQClient() override;

 protected:
  void InitTask() override;
  void Run() override;

 private:
  int mRunId;
  std::string mParameterName;
  std::string mOperationType;
  std::string mDataSource;
  std::string mObjectPath;
};
} // namespace ccdb
} // namespace o2

#endif /* ALICEO2_CDB_CONDITIONSMQCLIENT_H_ */
