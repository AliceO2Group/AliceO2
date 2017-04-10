/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Cave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef ALICEO2_PASSIVE_Cave_H
#define ALICEO2_PASSIVE_Cave_H

#include "FairModule.h"                 // for FairModule
#include "Rtypes.h"                     // for ClassDef, etc
namespace o2 {
namespace Passive {

class Cave : public FairModule
{
  public:
    Cave(const char* name, const char* Title="Exp Cave");
    Cave();
    ~Cave() override;
    void ConstructGeometry() override;

    /// Clone this object (used in MT mode only)
    FairModule* CloneModule() const override;

  private:
    Cave(const Cave& orig);
    Cave& operator=(const Cave&);

    Double_t mWorld[3];
    ClassDefOverride(o2::Passive::Cave,1) //
};
}
}
#endif //Cave_H
