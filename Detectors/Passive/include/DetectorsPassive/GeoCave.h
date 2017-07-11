// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


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


#ifndef ALICEO2_PASSIVE_GEOCAVE_H
#define ALICEO2_PASSIVE_GEOCAVE_H

#include "FairGeoSet.h"  // for FairGeoSet
#include "Rtypes.h"      // for GeoCave::Class, Bool_t, ClassDef, etc
#include "TString.h"     // for TString
#include <fstream>                      // for fstream

class FairGeoMedia;
namespace o2 {
namespace Passive {

class  GeoCave : public FairGeoSet
{
  protected:
    TString name;
  public:
    GeoCave();
    ~GeoCave() override = default;
    const char* getModuleName(Int_t) override {return name.Data();}
    Bool_t read(std::fstream&,FairGeoMedia*) override;
    void addRefNodes() override;
    void write(std::fstream&) override;
    void print() override;
    ClassDefOverride(o2::Passive::GeoCave,0) // Class for the geometry of CAVE
};
}
}
#endif  /* !PNDGEOCAVE_H */
