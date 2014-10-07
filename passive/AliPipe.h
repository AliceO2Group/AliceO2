/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliPipe file                                -----
// -----                Created by M. Al-Turany  June 2014             -----
// -------------------------------------------------------------------------

#ifndef PIPE_H
#define PIPE_H

#include "FairModule.h"

class AliPipe : public FairModule {
  public:
    AliPipe(const char * name, const char *Title="Ali Pipe");
    AliPipe();

    virtual ~AliPipe();
    virtual void ConstructGeometry();
   
  ClassDef(AliPipe,1) //AliPIPE

};

#endif //PIPE_H

