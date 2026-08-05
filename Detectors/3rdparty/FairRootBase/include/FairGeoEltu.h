/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOELTU_H
#define FAIRGEOELTU_H

#include "FairGeoBasicShape.h"   // for FairGeoBasicShape

#include <Rtypes.h>   // for FairGeoEltu::Class, Bool_t, etc
#include <iosfwd>     // for fstream

class FairGeoTransform;
class FairGeoVolume;
class TArrayD;

/**
 * class for geometry shape ELTU
 * @author Ilse  koenig
 */
class FairGeoEltu : public FairGeoBasicShape
{
  public:
    FairGeoEltu();
    ~FairGeoEltu();
    TArrayD* calcVoluParam(FairGeoVolume*);
    void calcVoluPosition(FairGeoVolume*, const FairGeoTransform&, const FairGeoTransform&);
    Int_t readPoints(std::fstream*, FairGeoVolume*);
    Bool_t writePoints(std::fstream*, FairGeoVolume*);
    void printPoints(FairGeoVolume* volu);
    ClassDef(FairGeoEltu, 0);
};

#endif /* !FAIRGEOELTU_H */
