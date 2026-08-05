/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOTRAP_H
#define FAIRGEOTRAP_H

#include "FairGeoBasicShape.h"   // for FairGeoBasicShape
#include "FairGeoRotation.h"     // for FairGeoRotation

#include <Rtypes.h>   // for FairGeoPcon::Class, Bool_t, etc

class FairGeoTransform;
class FairGeoVolume;
class TArrayD;

class FairGeoTrap : public FairGeoBasicShape
{
  private:
    FairGeoRotation intrinsicRot;

  public:
    FairGeoTrap();
    ~FairGeoTrap();
    TArrayD* calcVoluParam(FairGeoVolume*);
    void calcVoluPosition(FairGeoVolume*, const FairGeoTransform&, const FairGeoTransform&);
    ClassDef(FairGeoTrap, 0);   // class for geometry shape TRAP
};

#endif /* !FAIRGEOTRAP_H */
