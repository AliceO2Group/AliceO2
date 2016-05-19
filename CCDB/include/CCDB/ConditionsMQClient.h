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
 * @author R. Grosso, from examples/MQ/7-parameters/FairMQExample7Client.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQCLIENT_H_
#define ALICEO2_CDB_CONDITIONSMQCLIENT_H_

#include <string>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace CDB {

class ConditionsMQClient : public FairMQDevice
{
  public:
    enum
    {
        ParameterName = FairMQDevice::Last,
        Last
    };
    ConditionsMQClient();
    virtual ~ConditionsMQClient();

    static void CustomCleanup(void* data, void* hint);

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    virtual void Run();

  private:
    int fRunId;
    std::string fParameterName;
};
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQCLIENT_H_ */
