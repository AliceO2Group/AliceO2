/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef NEWDETECTORCONTFACT_H
#define NEWDETECTORCONTFACT_H

#include "FairContFact.h"

class FairContainer;

class AliItsContFact : public FairContFact
{
  private:
    void setAllContainers();
  public:
    AliItsContFact();
    ~AliItsContFact() {}
    FairParSet* createContainer(FairContainer*);
    ClassDef( AliItsContFact,0) // Factory for all AliIts parameter containers
};

#endif
