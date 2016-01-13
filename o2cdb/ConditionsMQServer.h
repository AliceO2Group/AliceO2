/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * ConditionsMQServer.h
 *
 * @since 2016-01-11
 * @author R. Grosso, from parmq/ParameterMQServer.h
 */

#ifndef ALICEO2_CDB_CONDITIONSMQSERVER_H_
#define ALICEO2_CDB_CONDITIONSMQSERVER_H_

#include <string>

#include "FairMQDevice.h"
#include "Manager.h"

class FairRuntimeDb;

namespace AliceO2 {
namespace CDB {

class ConditionsMQServer : public FairMQDevice
{
  public:
    enum
    {
        FirstInputName = FairMQDevice::Last,
        FirstInputType,
        SecondInputName,
        SecondInputType,
        OutputName,
        OutputType,
        Last
    };

    ConditionsMQServer();
    virtual ~ConditionsMQServer();

    virtual void Run();
    virtual void InitTask();

    static void CustomCleanup(void *data, void* hint);

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

  private:
    FairRuntimeDb* fRtdb;
    Manager* fCdbManager;

    std::string fFirstInputName;
    std::string fFirstInputType;
    std::string fSecondInputName;
    std::string fSecondInputType;
    std::string fOutputName;
    std::string fOutputType;
};
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQSERVER_H_ */
