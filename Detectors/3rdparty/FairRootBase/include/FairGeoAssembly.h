/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOASSEMBLY_H
#define FAIRGEOASSEMBLY_H

#include "FairGeoBasicShape.h"   // for FairGeoBasicShape

#include <Rtypes.h>   // for FairGeoAssembly::Class, etc
#include <iosfwd>     // for fstream

class FairGeoTransform;
class FairGeoVolume;
class TArrayD;

class FairGeoAssembly : public FairGeoBasicShape
{
  public:
    FairGeoAssembly();
    ~FairGeoAssembly();
    TArrayD* calcVoluParam(FairGeoVolume*);
    void calcVoluPosition(FairGeoVolume*, const FairGeoTransform&, const FairGeoTransform&);
    Int_t readPoints(std::fstream* pFile, FairGeoVolume* volu);
    Bool_t writePoints(std::fstream*, FairGeoVolume*);
    void printPoints(FairGeoVolume* volu);

    ClassDef(FairGeoAssembly, 0);   // class for geometry shape ASSEMBLY
};

#endif /* !FAIRGEOASSEMBLY_H */
