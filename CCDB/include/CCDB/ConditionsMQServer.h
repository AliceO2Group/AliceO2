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

#include "ParameterMQServer.h"
#include "CCDB/Manager.h"

namespace AliceO2 {
namespace CDB {

class ConditionsMQServer : public ParameterMQServer
{
  public:
    ConditionsMQServer();
    virtual ~ConditionsMQServer();

    virtual void Run();
    virtual void InitTask();

  private:
    Manager* fCdbManager;
};
}
}

#endif /* ALICEO2_CDB_CONDITIONSMQSERVER_H_ */
