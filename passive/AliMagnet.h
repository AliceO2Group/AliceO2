/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliMagnet  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef MAGNET_H
#define MAGNET_H

#include "FairModule.h"                 // for FairModule
#include "Rtypes.h"                     // for AliMagnet::Class, Bool_t, etc
#include <string>                       // for string

class AliMagnet : public FairModule
{
  public:
    AliMagnet(const char* name, const char* Title="MY Magnet");
    AliMagnet();
    virtual ~AliMagnet();
    void ConstructGeometry();
    ClassDef(AliMagnet,1)

};

#endif //MAGNET_H

