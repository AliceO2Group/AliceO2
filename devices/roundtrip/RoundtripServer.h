/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * RoundtripServer.h
 *
 * @since 2014-10-10
 * @author A. Rybalchenko
 */

#ifndef ROUNDTRIPSERVER_H_
#define ROUNDTRIPSERVER_H_

#include "FairMQDevice.h"

class RoundtripServer : public FairMQDevice
{
  public:
    RoundtripServer() {};
    virtual ~RoundtripServer() {};

  protected:
    virtual void Run();
};

#endif /* ROUNDTRIPSERVER_H_ */
