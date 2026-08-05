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
// FairGeoBasicShape
//
// Base class of the all shapes
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoBasicShape.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <fstream>     // for fstream
#include <iostream>    // for cout
#include <stdio.h>     // for printf, sprintf, sscanf
#include <string.h>    // for strlen

using std::cout;

ClassImp(FairGeoBasicShape);

FairGeoBasicShape::FairGeoBasicShape()
    : TNamed()
    , nPoints(0)
    , nParam(0)
    , param(0)
    , center(new FairGeoTransform())
    , position(new FairGeoTransform())
{
    // default constructor
}

FairGeoBasicShape::~FairGeoBasicShape()
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

Int_t FairGeoBasicShape::readPoints(std::fstream* pFile, FairGeoVolume* volu)
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
    Double_t x, y, z;
    const Int_t maxbuf = 155;
    Text_t buf[maxbuf];
    for (Int_t i = 0; i < nPoints; i++) {
        pFile->getline(buf, maxbuf);
        sscanf(buf, "%lf%lf%lf", &x, &y, &z);
        volu->setPoint(i, x, y, z);
    }
    return nPoints;
}

Bool_t FairGeoBasicShape::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes nPoints with 3 components to Ascii file
    if (!pFile) {
        return kFALSE;
    }
    Text_t buf[155];
    for (Int_t i = 0; i < volu->getNumPoints(); i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        sprintf(buf, "%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
        pFile->write(buf, strlen(buf));
    }
    return kTRUE;
}

void FairGeoBasicShape::printPoints(FairGeoVolume* volu)
{
    // prints nPoints with 3 components to screen
    for (Int_t i = 0; i < volu->getNumPoints(); i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        printf("%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
    }
}

void FairGeoBasicShape::calcVoluPosition(FairGeoVolume* volu, const FairGeoTransform& mTR)
{
    // calculates the relevant information to position the corresponding volume
    // in its mother and to position later other components inside this volume.
    // The transformation mTR describes the position and orientation of the
    // mother volume (center) relative to the physical coordinate system of
    // the volume from which it was created.
    FairGeoTransform& dTC = volu->getTransform();
    calcVoluPosition(volu, dTC, mTR);
}

void FairGeoBasicShape::posInMother(const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calculates the position of the volume inside its mother
    // dTC is the coordinate system of the ROOT volume relative to its physical
    // coordinate system
    // mTR is the coordinate system of the mother volume relative to its
    // physical coordinate system
    position->setRotMatrix(center->getRotMatrix());
    position->setTransVector(center->getTransVector());
    position->transFrom(dTC);
    position->transTo(mTR);
    FairGeoVector t(position->getTransVector());
    position->setTransVector(t *= 0.1);
}

void FairGeoBasicShape::printParam()
{
    // prints the parameters of the ROOT shape
    if (param) {
        for (Int_t i = 0; i < nParam; i++) {
            cout << param->At(i) << " ";
        }
        cout << '\n';
    }
}
