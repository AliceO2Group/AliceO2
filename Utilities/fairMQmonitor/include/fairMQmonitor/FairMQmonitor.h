/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * FairMQmonitor.h
 *
 * @since 2014-10-10
 * @author M. Krzewicki <mkrzewic@cern.ch>
 */

#ifndef FAIRMQMONITOR_H_
#define FAIRMQMONITOR_H_

#include "FairMQDevice.h"

class FairMQmonitor : public FairMQDevice
{
  public:
    FairMQmonitor();
    virtual ~FairMQmonitor();

  protected:
    virtual bool ConditionalRun();

};

#endif /* FAIRMQMONITOR_H_ */
