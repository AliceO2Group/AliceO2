/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : M. Al-Turany  21/04/2009

/////////////////////////////////////////////////////////////
//
// FairGeoAssmebly
//
// class for the shape Assembly
/////////////////////////////////////////////////////////////
#include "FairGeoAssembly.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>   // for TArrayD
#include <TString.h>   // for TString
#include <fstream>
#include <ostream>    // for fstream, etc
#include <stdio.h>    // for printf, sprintf
#include <string.h>   // for strlen

ClassImp(FairGeoAssembly);

FairGeoAssembly::FairGeoAssembly()
    : FairGeoBasicShape()
{
    // constructor
    fName = "ASSEMBLY";
    nPoints = 0;
    nParam = 0;
    param = new TArrayD(nParam);
}

FairGeoAssembly::~FairGeoAssembly()
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

Int_t FairGeoAssembly::readPoints(std::fstream*, FairGeoVolume*)
{

    // Assemblies has no parameters so just return 1;
    return 1;
}

TArrayD* FairGeoAssembly::calcVoluParam(FairGeoVolume*)
{
    // nothing to calculate

    return param;
}

Bool_t FairGeoAssembly::writePoints(std::fstream* pFile, FairGeoVolume* volu)
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

void FairGeoAssembly::printPoints(FairGeoVolume* volu)
{
    // prints volume points to screen
    for (Int_t i = 0; i < nPoints; i++) {
        FairGeoVector& v = *(volu->getPoint(i));
        printf("%9.3f\n", v(0));
    }
}

void FairGeoAssembly::calcVoluPosition(FairGeoVolume*, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    Double_t t[3] = {0., 0., 0.};
    center->setTransVector(t);
    posInMother(dTC, mTR);
}
