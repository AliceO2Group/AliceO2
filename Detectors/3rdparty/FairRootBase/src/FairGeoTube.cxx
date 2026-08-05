/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 11/11/2003 by Ilse Koenig
//*-- Modified : 28/06/99 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoTube
//
// class for the GEANT shape TUBE
//
// The size of a TUBE is defined by 3 'points'.
//   point 0:   origin of starting circle of the tube;
//   point 1:   inner radius of starting circle,
//              outer radius of starting circle;
//              (z-component not used)
//   point 2:   origin of ending circle of the tube;
// Warning: The x- and y-values of point 0 and 2 have to be the same!!!!
//          A rotation has to be desribed via the rotation matrix.
//
// The intrinsic coordinate system of a TUBE, which sits in the CAVE and is
// not rotated, is identical with the laboratory system.
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoTube.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>     // for TArrayD
#include <TMathBase.h>   // for Abs
#include <TString.h>     // for TString
#include <fstream>
#include <ostream>    // for basic_ostream::write
#include <stdio.h>    // for printf, sprintf, sscanf
#include <string.h>   // for strlen

ClassImp(FairGeoTube);

FairGeoTube::FairGeoTube()
    : FairGeoBasicShape()
{
    // constructor
    fName = "TUBE";
    nPoints = 3;
    nParam = 3;
    param = new TArrayD(nParam);
}

FairGeoTube::~FairGeoTube()
{
    // default destructor
    delete param;
    param = 0;
    delete center;
    center = 0;
    delete position;
    position = 0;
}

Int_t FairGeoTube::readPoints(std::fstream* pFile, FairGeoVolume* volu)
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
    Double_t x, y, z;
    const Int_t maxbuf = 155;
    Text_t buf[maxbuf];
    for (Int_t i = 0; i < nPoints; i++) {
        pFile->getline(buf, maxbuf);
        if (i != 1) {
            sscanf(buf, "%lf%lf%lf", &x, &y, &z);
            volu->setPoint(i, x, y, z);
        } else {
            sscanf(buf, "%lf%lf", &x, &y);
            volu->setPoint(i, x, y, 0.0);
        }
    }
    return nPoints;
}

Bool_t FairGeoTube::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes the 3 'points' decribed above to ascii file
    if (!pFile) {
        return kFALSE;
    }
    Text_t buf[155];
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        if (i != 1) {
            sprintf(buf, "%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
        } else {
            sprintf(buf, "%9.3f%10.3f\n", v(0), v(1));
        }
        pFile->write(buf, strlen(buf));
    }
    return kTRUE;
}

void FairGeoTube::printPoints(FairGeoVolume* volu)
{
    // prints volume points to screen
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        if (i != 1) {
            printf("%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
        } else {
            printf("%9.3f%10.3f\n", v(0), v(1));
        }
    }
}

TArrayD* FairGeoTube::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape TUBE
    Double_t fac = 10.;
    FairGeoVector& v1 = *(volu->getPoint(1));
    param->AddAt(v1(0) / fac, 0);
    param->AddAt(v1(1) / fac, 1);
    FairGeoVector v = *(volu->getPoint(2)) - *(volu->getPoint(0));
    param->AddAt(TMath::Abs(v(2)) / fac / 2., 2);
    return param;
}

void FairGeoTube::calcVoluPosition(FairGeoVolume* volu, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calculates the position of the center of the ROOT volume in the intrinsic
    // coordinate system and stores it in the data element 'center'
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    Double_t t[3] = {0., 0., 0.};
    FairGeoVector v = *(volu->getPoint(2)) + *(volu->getPoint(0));
    t[2] = v(2) / 2.;
    center->clear();
    center->setTransVector(t);
    posInMother(dTC, mTR);
}
