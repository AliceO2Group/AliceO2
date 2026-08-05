/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 27/05/99

/////////////////////////////////////////////////////////////
// FairGeoVolume
//
// Class to hold the basic geometry properties of a volume
// Each volume has a name, a pointer to a reference volume and
// a lab transformation.
// The reference volume defines the shape, the mother, the size
// and the transformation relative to the mother volume which
// is either the cave (for modules) or the detector (for the
// inner parts).
// As an example:
// The Mdc modules built at GSI (plane 1 type) are all identical
// independent where the sit in the cave. This module type has a
// fixed coordinate system. The first layers in all these modules
// are identical and they have the same position in the module
// coordinate system.
//
/////////////////////////////////////////////////////////////
#include "FairGeoVolume.h"

#include <iostream>   // for operator<<, basic_ostream, etc

using std::cout;
using std::endl;

ClassImp(FairGeoVolume);

FairGeoVolume::FairGeoVolume(FairGeoVolume& r)
    : TNamed(r)
    , shape("")
    , mother("")
    , points(NULL)
    , transform(FairGeoTransform())
    , fLabTransform(FairGeoTransform())
    , fMedium(0)
    , nPoints(0)
    , fHadFormat(0)
    , fgMCid(0)
{
    // copy constructor
    //  fName=r.GetName();
    setVolumePar(r);
}

void FairGeoVolume::setVolumePar(FairGeoVolume& r)
{
    // copies all volume parameters except the name
    shape = r.getShape();
    mother = r.getMother();
    Int_t n = r.getNumPoints();
    createPoints(n);
    for (Int_t i = 0; i < nPoints; i++) {
        setPoint(i, *(r.getPoint(i)));
    }
    transform = r.getTransform();
}

void FairGeoVolume::createPoints(const Int_t n)
{
    // Creates n Points (objects of class FairGeoVector).
    // If the array exists already and the size is different from n it is
    // deleted and recreated with the new size n.
    // If n==0 the points are deleted.
    if (n != nPoints) {
        nPoints = n;
        if (n > 0) {
            if (points != 0) {
                points->Delete();
                delete points;
            }
            points = new TObjArray(n);
            for (Int_t i = 0; i < n; i++) {
                points->AddAt(new FairGeoVector(), i);
            }
        } else {
            if (points) {
                points->Delete();
            }
            delete points;
            points = 0;
        }
    }
}

void FairGeoVolume::setPoint(const Int_t n, const Double_t x, const Double_t y, const Double_t z)
{
    // set the 3 values of the point with index n
    if (points && n < nPoints) {
        FairGeoVector* v = static_cast<FairGeoVector*>(points->At(n));
        v->setX(x);
        v->setY(y);
        v->setZ(z);
    }
}

void FairGeoVolume::setPoint(const Int_t n, const FairGeoVector& p)
{
    // sets point with index n by copying the 3 components of point p
    if (points && n < nPoints) {
        FairGeoVector& v = *(static_cast<FairGeoVector*>(points->At(n)));
        v = p;
    }
}

void FairGeoVolume::clear()
{
    // clears the volume
    // deletes the points
    shape = "";
    mother = "";
    if (points) {
        points->Delete();
    }
    delete points;
    points = 0;
    nPoints = 0;
    transform.clear();
}

void FairGeoVolume::print()
{
    // prints all parameters of a volume
    cout << "Volume: " << (const_cast<const char*>(fName.Data()))
         << "  Shape: " << (const_cast<const char*>(shape.Data()))
         << "  Mother: " << (const_cast<const char*>(mother.Data())) << '\n';
    cout << "Points definition " << endl;
    if (points) {
        for (Int_t i = 0; i < nPoints; i++) {
            cout << (*(static_cast<FairGeoVector*>(points->At(i))));
        }
    }
    cout << "Lab Transform " << endl;
    fLabTransform.print();
    cout << '\n';
}

Double_t FairGeoVolume::getVolParameter(Int_t nPoint, Int_t pos)
{
    FairGeoVector* vec = static_cast<FairGeoVector*>(points->At(nPoint));
    if (vec) {
        return vec->getValues(pos);
    } else {
        return -1;
    }
}

FairGeoVolume::~FairGeoVolume()
{
    if (points) {
        points->Delete();
        delete points;
        points = 0;
    }
}
