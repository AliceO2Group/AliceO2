/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliPassiveContFact  file                    -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef ALICEO2_PASSIVE_CONTFACT_H
#define ALICEO2_PASSIVE_CONTFACT_H

#include "FairContFact.h"               // for FairContFact, etc
#include "Rtypes.h"                     // for AliPassiveContFact::Class, etc

class FairParSet;

namespace o2 {
namespace Passive {

class PassiveContFact : public FairContFact
{
  private:
    void setAllContainers();
  public:
    PassiveContFact();
    ~PassiveContFact() override {;}
    FairParSet* createContainer(FairContainer*) override;
    ClassDefOverride(o2::Passive::PassiveContFact,0) // Factory for all Passive parameter containers
};
}
}
#endif  /* !PNDPASSIVECONTFACT_H */
