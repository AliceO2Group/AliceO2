/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

/*
 * FairSystemInfo.h
 *
 *  Created on: Mai 18, 2015
 *      Author: f.uhlig
 */

#ifndef BASE_FAIRSYSTEMINFO_H_
#define BASE_FAIRSYSTEMINFO_H_

#include <Rtypes.h>
#include <cstddef>

class FairSystemInfo
{
  public:
    FairSystemInfo() {}
    virtual ~FairSystemInfo() {}

    Float_t GetMaxMemory();
    size_t GetCurrentMemory();

    ClassDef(FairSystemInfo, 1);
};

#endif   // BASE_FAIRSYSTEMINFO_H_
