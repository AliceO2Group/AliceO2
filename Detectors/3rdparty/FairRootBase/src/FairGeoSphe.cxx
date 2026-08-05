/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 11/11/2003 by Ilse Koenig
//*-- Modified : 29/06/99 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoSphe
//
// class for the GEANT shape SPHE
//
// The size of a SPHE is defined by 3 'points' (The z-components are not used)
//   point 0:   inner radius of the shell
//              outer radius of the shell;
//   point 1:   starting polar angle of the shell,
//              ending polar angle of the shell;
//   point 2:   starting azimuthal angle of the shell,
//              ending azimuthal angle of the shell; )
//
// The intrinsic coordinate system of a SPHE, which sits in the CAVE and is
// not rotated, is identical with the laboratory system.
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoSphe.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <TString.h>   // for TString
#include <fstream>
#include <ostream>    // for basic_ostream::write
#include <stdio.h>    // for printf, sprintf, sscanf
#include <string.h>   // for strlen

ClassImp(FairGeoSphe);

FairGeoSphe::FairGeoSphe()
    : FairGeoBasicShape()
{
    // constructor
    fName = "SPHE";
    nPoints = 3;
    nParam = 6;
    param = new TArrayD(nParam);
}

FairGeoSphe::~FairGeoSphe()
{
    // default destructor
    delete param;
    param = 0;
    delete center;
    center = 0;
    delete position;
    position = 0;
}

Int_t FairGeoSphe::readPoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // reads the 3 'points' decribed above from ascii file
    // if the array of points is not existing in the volume it is created and
    // the values are stored inside
    // returns the number of points
    if (!pFile) {
        return 0;
    }
    if (volu->getNumPoints() != nPoints) {
        volu->createPoints(nPoints);
    }
    Double_t x, y;
    const Int_t maxbuf = 155;
    Text_t buf[maxbuf];
    for (Int_t i = 0; i < nPoints; i++) {
        pFile->getline(buf, maxbuf);
        sscanf(buf, "%lf%lf", &x, &y);
        volu->setPoint(i, x, y, 0.0);
    }
    return nPoints;
}

Bool_t FairGeoSphe::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes the 3 'points' decribed above to ascii file
    if (!pFile) {
        return kFALSE;
    } else {
        Text_t buf[155];
        for (Int_t i = 0; i < nPoints; i++) {
            FairGeoVector& v = *(volu->getPoint(i));
            sprintf(buf, "%9.3f%10.3f\n", v(0), v(1));
            pFile->write(buf, strlen(buf));
        }
        return kTRUE;
    }
}

void FairGeoSphe::printPoints(FairGeoVolume* volu)
{
    // prints volume points to screen
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        printf("%9.3f%10.3f\n", v(0), v(1));
    }
}

TArrayD* FairGeoSphe::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape SPHE
    Double_t fac = 10.;
    FairGeoVector& v0 = *(volu->getPoint(0));
    FairGeoVector& v1 = *(volu->getPoint(1));
    FairGeoVector& v2 = *(volu->getPoint(2));
    param->AddAt(v0(0) / fac, 0);
    param->AddAt(v0(1) / fac, 1);
    param->AddAt(v1(0), 2);
    param->AddAt(v1(1), 3);
    param->AddAt(v2(0), 4);
    param->AddAt(v2(1), 5);
    return param;
}

void FairGeoSphe::calcVoluPosition(FairGeoVolume*, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    center->clear();
    posInMother(dTC, mTR);
}
