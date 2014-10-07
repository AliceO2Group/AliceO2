
/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliGeoCave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef MYGEOCAVE_H
#define MYGEOCAVE_H

#include "FairGeoSet.h"                 // for FairGeoSet
#include "Riosfwd.h"                    // for fstream
#include "Rtypes.h"                     // for AliGeoCave::Class, Bool_t, etc
#include "TString.h"                    // for TString

#include <fstream>                      // for fstream

class FairGeoMedia;

class  AliGeoCave : public FairGeoSet
{
  protected:
    TString name;
  public:
    AliGeoCave();
    ~AliGeoCave() {}
    const char* getModuleName(Int_t) {return name.Data();}
    Bool_t read(fstream&,FairGeoMedia*);
    void addRefNodes();
    void write(fstream&);
    void print();
    ClassDef(AliGeoCave,0) // Class for the geometry of CAVE
};

#endif  /* !PNDGEOCAVE_H */
