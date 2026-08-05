/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOTRD1_H
#define FAIRGEOTRD1_H

#include "FairGeoBasicShape.h"   // for FairGeoBasicShape
#include "FairGeoRotation.h"     // for FairGeoRotation

#include <Rtypes.h>   // for FairGeoPcon::Class, Bool_t, etc

class FairGeoTransform;
class FairGeoVolume;
class TArrayD;

class FairGeoTrd1 : public FairGeoBasicShape
{
  private:
    FairGeoRotation intrinsicRot;

  public:
    FairGeoTrd1();
    ~FairGeoTrd1();
    TArrayD* calcVoluParam(FairGeoVolume*);
    void calcVoluPosition(FairGeoVolume*, const FairGeoTransform&, const FairGeoTransform&);

    ClassDef(FairGeoTrd1, 0);   // class for geometry shape TRD1
};

#endif /* !FAIRGEOTRD1_H */
