/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliCave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef Cave_H
#define Cave_H

#include "FairModule.h"                 // for FairModule

#include "Rtypes.h"                     // for AliCave::Class, ClassDef, etc

class AliCave : public FairModule
{
  public:
    AliCave(const char* name, const char* Title="Exp Cave");
    AliCave();
    virtual ~AliCave();
    virtual void ConstructGeometry();


  private:
    Double_t world[3];
    ClassDef(AliCave,1) //PNDCaveSD
};

#endif //Cave_H

