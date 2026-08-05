/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 11/11/2003 by Ilse Koenig
//*-- Modified : 26/11/2001 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoPcon
//
// class for the GEANT shape PCON
//
// The size of a PCON is defined by a variable number of 'points'.
//   point 0:   number of planes perpendicular to the z axis where the
//              dimension of the section is given;
//   point 1:   azimutal angle phi at which the volume begins,
//              opening angle dphi of the volume,
//              (z-component not used)
//   point 2ff: z coordinate of the section,
//              inner radius at position z,
//              outer radius at position z;
//
// The intrinsic coordinate system of a PCON, which sits in the CAVE and is
// not rotated, is identical with the laboratory system.
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoPcon.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <TString.h>   // for TString
#include <fstream>
#include <ostream>    // for fstream, etc
#include <stdio.h>    // for printf, sprintf, sscanf
#include <string.h>   // for strlen

ClassImp(FairGeoPcon);

FairGeoPcon::FairGeoPcon()
    : FairGeoBasicShape()
{
    // constructor
    fName = "PCON";
    nPoints = 0;
    nParam = 0;
}

FairGeoPcon::~FairGeoPcon()
{
    // default destructor
    delete param;
    param = 0;
    delete center;
    center = 0;
    delete position;
    position = 0;
}

Int_t FairGeoPcon::readPoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // reads the 'points' decribed above from ascii file and stores them in the
    // array 'points' of the volume
    // returns the number of points
    if (!pFile) {
        return 0;
    }
    Double_t x, y, z;
    const Int_t maxbuf = 155;
    Text_t buf[maxbuf];
    pFile->getline(buf, maxbuf);
    Int_t n;
    sscanf(buf, "%i", &n);
    if (n <= 0) {
        return 0;
    }
    nPoints = n + 2;
    if (volu->getNumPoints() != nPoints) {
        volu->createPoints(nPoints);
    }
    volu->setPoint(0, static_cast<Double_t>(n), 0.0, 0.0);
    for (Int_t i = 1; i < nPoints; i++) {
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

Bool_t FairGeoPcon::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes the 'points' decribed above to ascii file
    if (!pFile) {
        return kFALSE;
    }
    Text_t buf[155];
    for (Int_t i = 0; i < volu->getNumPoints(); i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        switch (i) {
            case 0:
                sprintf(buf, "%3i\n", static_cast<Int_t>(v(0)));
                break;
            case 1:
                sprintf(buf, "%9.3f%10.3f\n", v(0), v(1));
                break;
            default:
                sprintf(buf, "%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
        }
        pFile->write(buf, strlen(buf));
    }
    return kTRUE;
}

void FairGeoPcon::printPoints(FairGeoVolume* volu)
{
    // prints volume points to screen
    for (Int_t i = 0; i < volu->getNumPoints(); i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        switch (i) {
            case 0:
                printf("%3i\n", static_cast<Int_t>(v(0)));
                break;
            case 1:
                printf("%9.3f%10.3f\n", v(0), v(1));
                break;
            default:
                printf("%9.3f%10.3f%10.3f\n", v(0), v(1), v(2));
        }
    }
}

TArrayD* FairGeoPcon::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape PCON
    Double_t fac = 10.;
    nPoints = volu->getNumPoints();
    nParam = (nPoints - 1) * 3;
    if (param && param->GetSize() != nParam) {
        delete param;
        param = 0;
    }
    if (!param) {
        param = new TArrayD(nParam);
    }
    FairGeoVector& v1 = *(volu->getPoint(1));
    Int_t k = 0;
    param->AddAt(v1(0), k++);
    param->AddAt(v1(1), k++);
    param->AddAt((nPoints - 2), k++);
    for (Int_t i = 2; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        param->AddAt(v(0) / fac, k++);
        param->AddAt(v(1) / fac, k++);
        param->AddAt(v(2) / fac, k++);
    }
    return param;
}

void FairGeoPcon::calcVoluPosition(FairGeoVolume*, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    center->clear();
    posInMother(dTC, mTR);
}
