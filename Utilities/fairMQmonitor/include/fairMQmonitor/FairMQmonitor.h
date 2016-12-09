/********************************************************************************
 *    Copyright (C) 2014 Frankfurt Institute for Advanced Studies               *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * FairMQmonitor.h
 *
 * @since 2014-12-10
 * @author M. Krzewicki <mkrzewic@cern.ch>
 */

#ifndef FAIRMQMONITOR_H_
#define FAIRMQMONITOR_H_

#include "O2device/O2device.h"

class FairMQmonitor : public AliceO2::Base::O2device
{
  public:
    FairMQmonitor();
    virtual ~FairMQmonitor();

  protected:
    virtual bool ConditionalRun();
    bool HandleData(FairMQParts& parts, int index);
    bool HandleBuffers(byte* headerBuffer, size_t headerBufferSize,
                       byte* dataBuffer,   size_t dataBufferSize);

};

#endif /* FAIRMQMONITOR_H_ */
