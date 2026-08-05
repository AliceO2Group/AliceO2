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
// FairGeoTrap
//
// class for the GEANT shape TRAP
//
// The technical coordinate system of a TRAP, which sits in
// CAVE and is not rotated, is the laboratory system.
// The y-axis points from the smaller side to the larger one.
// That's the same definitition as for a TRAP and different from
// the Geant or ROOT definition for a TRAP.
// Therefore a transformation is needed:
//              x-technical = - (x-Geant)
//              y-technical = - (y-Geant)
//              z-technical = z-Geant
// This is stored in the data element intrinsicRot which is
// created in the function calcVoluPosition(...)
//
/////////////////////////////////////////////////////////////
#include "FairGeoTrap.h"

#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector
#include "FairGeoVolume.h"      // for FairGeoVolume

#include <TArrayD.h>     // for TArrayD
#include <TMath.h>       // for ATan, Pi, Sqrt, atan
#include <TMathBase.h>   // for Abs
#include <TString.h>     // for TString
#include <iostream>      // for operator<<, cout, etc

using std::cout;

ClassImp(FairGeoTrap);

FairGeoTrap::FairGeoTrap()
    : FairGeoBasicShape()
    , intrinsicRot(FairGeoRotation())
{
    // constructor
    fName = "TRAP";
    nPoints = 8;
    nParam = 11;
    param = new TArrayD(nParam);
    intrinsicRot.setElement(-1., 0);
    intrinsicRot.setElement(-1., 4);
}

FairGeoTrap::~FairGeoTrap()
{
    // destructor
    delete param;
    param = 0;
    delete center;
    center = 0;
    delete position;
    position = 0;
}

TArrayD* FairGeoTrap::calcVoluParam(FairGeoVolume* volu)
{
    // calculates the parameters needed to create the shape
    const Double_t fac = 20.;
    const Double_t raddeg = 180. / TMath::Pi();
    Double_t alpha, beta;
    FairGeoVector cb, ct, dc;
    for (Int_t i = 0; i < 4; i++) {
        cb += *(volu->getPoint(i));
    }   // bottom plane
    for (Int_t j = 4; j < 8; j++) {
        ct += *(volu->getPoint(j));
    }   // top plane
    dc = (ct - cb);
    //  cout << dc <<" "<< ct <<" "<< cb <<" "<< endl;
    dc *= 0.25;        // vector from bottom to top plane
    dc.setX(-dc(0));   // GEANT coordinate system
    dc.setY(-dc(1));
    param->AddAt(TMath::Abs(dc(2)) / fac, 0);
    alpha = TMath::ATan(TMath::Sqrt(dc(0) * dc(0) + dc(1) * dc(1)) / dc(2)) * raddeg;
    if (TMath::Abs(alpha) < 0.0001) {
        alpha = 0.0;
        beta = 0.0;
    } else {
        if (TMath::Abs(dc(0)) < 0.0001) {
            if (dc(1) > 0) {
                beta = 90.0;
            } else {
                beta = 270.0;
            }
        } else {
            beta = atan(dc(1) / dc(0)) * raddeg;
            if (dc(0) < 0) {
                beta = 180.0 + beta;
            }
            if (beta < 0) {
                beta = 360.0 + beta;
            }
        }
    }
    param->AddAt(alpha, 1);
    param->AddAt(beta, 2);
    param->AddAt(((*(volu->getPoint(1)))(1) - (*(volu->getPoint(0)))(1)) / fac, 3);
    param->AddAt(((*(volu->getPoint(1)))(0) - (*(volu->getPoint(2)))(0)) / fac, 4);
    param->AddAt(((*(volu->getPoint(0)))(0) - (*(volu->getPoint(3)))(0)) / fac, 5);

    Double_t a = TMath::ATan(((*(volu->getPoint(1)))(0) - (*(volu->getPoint(0)))(0) + (*(volu->getPoint(2)))(0)
                              - (*(volu->getPoint(3)))(0))
                             / 40. / param->At(3))
                 * raddeg;
    if (TMath::Abs(a) <= 0.0001) {
        param->AddAt(0., 6);
    } else {
        param->AddAt(a, 6);
    }
    param->AddAt(((*(volu->getPoint(5)))(1) - (*(volu->getPoint(4)))(1)) / fac, 7);
    param->AddAt(((*(volu->getPoint(5)))(0) - (*(volu->getPoint(6)))(0)) / fac, 8);
    param->AddAt(((*(volu->getPoint(4)))(0) - (*(volu->getPoint(7)))(0)) / fac, 9);
    a = TMath::ATan(((*(volu->getPoint(5)))(0) - (*(volu->getPoint(4)))(0) + (*(volu->getPoint(6)))(0)
                     - (*(volu->getPoint(7)))(0))
                    / 40. / param->At(7))
        * raddeg;
    if (TMath::Abs(a) <= 0.0001) {
        param->AddAt(0., 10);
    } else {
        param->AddAt(a, 10);
    }
    // check if coplanar
    Double_t dx = (param->At(4) - param->At(5)) / param->At(3) * param->At(7) - (param->At(8) - param->At(9));
    if (TMath::Abs(dx) >= 0.001) {
        cout << "top and bottom plane are not coplanar for shape TRAP\n";
        cout << "lenght in x-direction of top plane is changed\n";
        cout << "old values: " << param->At(8) << "  " << param->At(9) << "\n";
        param->AddAt(param->At(8) + dx / 2., 8);
        param->AddAt(param->At(9) - dx / 2., 9);
        cout << "new values: " << param->At(8) << "  " << param->At(9) << "\n";
    }
    return param;
}

void FairGeoTrap::calcVoluPosition(FairGeoVolume* volu, const FairGeoTransform& dTC, const FairGeoTransform& mTR)
{
    // calls the function posInMother(...) to calculate the position of the
    // volume in its mother
    Double_t t[3] = {0., 0., 0.};
    for (Int_t i = 0; i < 8; i++) {
        t[0] += (*(volu->getPoint(i)))(0);
    }
    t[0] /= 8.;
    t[1] =
        ((*(volu->getPoint(1)))(1) + (*(volu->getPoint(0)))(1) + (*(volu->getPoint(5)))(1) + (*(volu->getPoint(4)))(1))
        / 4.;
    t[2] = ((*(volu->getPoint(4)))(2) + (*(volu->getPoint(0)))(2)) / 2.;
    center->setTransVector(t);
    center->setRotMatrix(intrinsicRot);
    posInMother(dTC, mTR);
}
