/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 11/11/2003 by Ilse Koenig
//*-- Modified : 14/05/99 by Ilse Koenig

/////////////////////////////////////////////////////////////
//
// FairGeoBrik
//
// class for the GEANT shape BOX
//
// The intrisic coordinate system of a BOX in GEANT/ROOT has the
// same orientation as the volume described by its 8 points
// in the technical coordinate system
//
/////////////////////////////////////////////////////////////
#include "FairGeoBrik.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <TString.h>   // for TString

ClassImp(FairGeoBrik);

FairGeoBrik::FairGeoBrik()
    : FairGeoBasicShape()
{
    // constructor
    fName = "BOX ";
    nPoints = 8;
    nParam = 3;
    param = new TArrayD(nParam);
}

FairGeoBrik::~FairGeoBrik()
{
    // destructor
    if (param) {
        delete param;
        param = 0;
    }
    if (center) {
        delete center;
        center = 0;
    }
    if (position) {
        delete position;
        position = 0;
    }
}

TArrayD* FairGeoBrik::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape
    if (!volu) {
        return 0;
    }
    FairGeoVector v = *(volu->getPoint(5)) - *(volu->getPoint(3));
    v.abs();
    v *= (1 / 20.);
    for (Int_t i = 0; i < nParam; i++) {
        param->AddAt(v(i), i);
    }
    return param;
}

void FairGeoBrik::calcVoluPosition(FairGeoVolume* volu, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    FairGeoVector trans = *(volu->getPoint(5)) + *(volu->getPoint(3));
    trans *= 0.5;
    center->clear();
    center->setTransVector(trans);
    posInMother(dTC, mTR);
}
