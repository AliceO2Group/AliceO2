/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 11/11/2003 by Ilse Koenig
//*-- Modified : 16/05/99 by Ilse Koenig

/////////////////////////////////////////////////////////////
//
// FairGeoTrd1
//
// class for the GEANT shape TRD1
//
// The technical coordinate system of a TRD1, which sits in
// CAVE and is not rotated, is the laboratory system.
// The y-axis points from the smaller side to the larger one.
// That's the same definitition as for a TRAP and different from
// the Geant or ROOT definition for a TRD1.
// Therefore a transformation is needed:
//              x-technical = - (x-Geant)
//              y-technical = z-Geant
//              z-technical = y-Geant
// This is stored in the data element intrinsicRot which is
// created in the function calcVoluPosition(...)
//
/////////////////////////////////////////////////////////////
#include "FairGeoTrd1.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>     // for TArrayD
#include <TMathBase.h>   // for Abs
#include <TString.h>     // for TString

ClassImp(FairGeoTrd1);

FairGeoTrd1::FairGeoTrd1()
    : FairGeoBasicShape()
    , intrinsicRot(FairGeoRotation())
{
    // constructor
    fName = "TRD1";
    nPoints = 8;
    nParam = 4;
    param = new TArrayD(nParam);
    intrinsicRot.setZero();
    intrinsicRot.setElement(-1., 0);
    intrinsicRot.setElement(1., 5);
    intrinsicRot.setElement(1., 7);
}

FairGeoTrd1::~FairGeoTrd1()
{
    // destructor
    delete param;
    param = 0;
    delete center;
    center = 0;
    delete position;
    position = 0;
}

TArrayD* FairGeoTrd1::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape
    Double_t fac = 20.;
    FairGeoVector& v0 = *(volu->getPoint(0));
    FairGeoVector& v1 = *(volu->getPoint(1));
    FairGeoVector& v2 = *(volu->getPoint(2));
    FairGeoVector& v3 = *(volu->getPoint(3));
    FairGeoVector& v4 = *(volu->getPoint(4));
    param->AddAt(TMath::Abs((v0(0) - v3(0)) / fac), 0);
    param->AddAt((v1(0) - v2(0)) / fac, 1);
    param->AddAt((v4(2) - v0(2)) / fac, 2);
    param->AddAt((v1(1) - v0(1)) / fac, 3);
    return param;
}

void FairGeoTrd1::calcVoluPosition(FairGeoVolume* volu, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    Double_t t[3] = {0., 0., 0.};
    for (Int_t i = 0; i < 4; i++) {
        t[0] += (*(volu->getPoint(i)))(0);
    }
    t[0] /= 4.;
    FairGeoVector& v0 = *(volu->getPoint(0));
    FairGeoVector& v1 = *(volu->getPoint(1));
    FairGeoVector& v4 = *(volu->getPoint(4));
    t[1] = (v1(1) + v0(1)) / 2.;
    t[2] = (v4(2) + v0(2)) / 2.;
    center->setTransVector(t);
    center->setRotMatrix(intrinsicRot);
    posInMother(dTC, mTR);
}
