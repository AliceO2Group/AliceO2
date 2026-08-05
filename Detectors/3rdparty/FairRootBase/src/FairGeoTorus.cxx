/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : M. Al-Turany  06/11/2006

/////////////////////////////////////////////////////////////
//
// FairGeoTorus
//
// class for the shape Torus
// A torus has 5 parameters :
//            R    - axial radius
//            Rmin - inner radius
//            Rmax - outer radius
//            Phi1 - starting phi
//            Dphi - phi extent
/////////////////////////////////////////////////////////////
#include "FairGeoTorus.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <TString.h>   // for TString
#include <fstream>
#include <ostream>    // for basic_ostream::write
#include <stdio.h>    // for printf, sprintf, sscanf
#include <string.h>   // for strlen

ClassImp(FairGeoTorus);

FairGeoTorus::FairGeoTorus()
    : FairGeoBasicShape()
{
    // constructor
    fName = "TORUS";
    nPoints = 5;
    nParam = 5;
    param = new TArrayD(nParam);
}

FairGeoTorus::~FairGeoTorus()
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

Int_t FairGeoTorus::readPoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // reads nPoints with 3 components from Ascii file
    // if the array of points is not existing in the volume it is created and
    // the values are stored inside
    // returns the number of points
    if (!pFile) {
        return 0;
    }
    if (volu->getNumPoints() != nPoints) {
        volu->createPoints(nPoints);
    }
    Double_t x;
    const Int_t maxbuf = 155;
    Text_t buf[maxbuf];
    for (Int_t i = 0; i < nPoints; i++) {
        pFile->getline(buf, maxbuf);
        sscanf(buf, "%lf", &x);
        volu->setPoint(i, x, 0, 0);
    }
    return nPoints;
}

TArrayD* FairGeoTorus::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape

    for (Int_t i = 0; i < 3; i++) {
        FairGeoVector v = *(volu->getPoint(i));
        v *= (1 / 10.);
        param->AddAt(v(0), i);
    }
    FairGeoVector v = *(volu->getPoint(3));
    param->AddAt(v(0), 3);

    v = *(volu->getPoint(4));
    param->AddAt(v(0), 4);

    return param;
}

Bool_t FairGeoTorus::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes the 4 'points' decribed above to ascii file
    if (!pFile) {
        return kFALSE;
    }
    Text_t buf[155];
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        sprintf(buf, "%9.3f\n", v(0));
        pFile->write(buf, strlen(buf));
    }
    return kTRUE;
}

void FairGeoTorus::printPoints(FairGeoVolume* volu)
{
    // prints volume points to screen
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        printf("%9.3f\n", v(0));
    }
}

void FairGeoTorus::calcVoluPosition(FairGeoVolume*, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    Double_t t[3] = {0., 0., 0.};
    center->setTransVector(t);
    posInMother(dTC, mTR);
}
