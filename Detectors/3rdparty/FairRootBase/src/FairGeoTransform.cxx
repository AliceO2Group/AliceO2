/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 16/06/1999
//*-- Modified : 21/06/2005 D.Bertini

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoTransform
//
// Class to hold the orientation (rotation matrix) and the position
// (translation vector) of a coordinate system (system 2) relative to a
// reference coordinate system (system 1)
// It provides member functions to transform a vector or a point and an other
// coordinate system from its own coordinate system into the reference
// coordinate system and vice versa.
// Instances of this class can e.g. hold the lab or detector transformation of
// a geometry volume (see class FairGeoVolume)
//
// Inline functions:
//
// FairGeoTransform()
//    The default constructor creates an identity transformation
// FairGeoTransform(FairGeoTransform& t)
//    copy constructor
// void setTransform(const FairGeoTransform& t)
//    copies the given transformation
// void setRotMatrix(const FairGeoRotation& r)
//    copies the given rotation matrix
// void setRotMatrix(const Double_t* a)
//    creates a rotation matrix taking an Double_t array with 9 components
// void setTransVector(const FairGeoVector& t)
//    copies the given translation vector
// void setTransVector(const Double_t* a)
//    creates a translation vector taking an Double_t array with 6 components
// const FairGeoRotation& getRotMatrix() const
//    returns the rotation matrix
// const FairGeoVector& getTransVector() const
//    returns the  translation vector
//
///////////////////////////////////////////////////////////////////////////////

#include "FairGeoTransform.h"

//#include <iostream>
//#include <iomanip>
//#include "math.h"

ClassImp(FairGeoTransform);

FairGeoTransform::FairGeoTransform()
    : TObject()
    , rot(FairGeoRotation(0, 0, 0))
    , trans(FairGeoVector(0, 0, 0))
    , trans_cm(FairGeoVector(0, 0, 0))
{}

FairGeoTransform& FairGeoTransform::operator=(const FairGeoTransform& t)
{
    rot = t.getRotMatrix();
    trans = t.getTransVector();

    return *this;
}

FairGeoVector FairGeoTransform::transFrom(const FairGeoVector& p) const
{
    // Transforms a vector (point) given in its own coordinate
    // system (2) into the reference coordinate system (1)
    // e.g. v2 is a vector (point) in the detector coordinate system;
    // it can be transformed to a vector v2 the lab system with
    //    FairGeoVector v2=mo.transFrom(v1)
    // where mo is the coordinate system of the mother
    return rot * p + trans;
}

FairGeoVector FairGeoTransform::transTo(const FairGeoVector& p) const
{
    // Transforms a vector (point) given in the reference system (1)
    // into the local coordinate system (2)
    // e.g. v1 is a vector (point) in the lab system; it can be transformed to
    // a vector v2 the detector coordinate system with
    //    FairGeoVector v2=mo.transTo(v1)
    // where mo is the coordinate system of the mother
    return rot.inverse() * (p - trans);
}

void FairGeoTransform::transTo(const FairGeoTransform& s)
{
    // Transforms the coordinate system into the coordinate system
    // described by s. Both transformations must have the same reference
    // system e.g. the lab system
    // This function is e.g. used to transform a daughter coordinate system
    // with a transformation relative to the lab into the detector coordinate
    // system.
    const FairGeoRotation& rm = s.getRotMatrix();
    FairGeoRotation rt(rm.inverse());
    if (rm.diff2(rot) < 0.000001) {
        rot.setUnitMatrix();
    } else {
        rot.transform(rt);
    }
    trans -= s.getTransVector();
    trans = rt * trans;
    //  trans.round(3); // rounds to 3 digits (precision 1 micrometer)
}

void FairGeoTransform::transFrom(const FairGeoTransform& s)
{
    // Transforms the coordinate system described by s into the local
    // coordinate system
    // This function is e.g. used to transform a daughter coordinate system
    // with a transformation relative to its mother into the lab system.
    // e.g. daughterDetTransform.transFrom(motherLabTransform)
    const FairGeoRotation& r = s.getRotMatrix();
    rot.transform(r);
    trans = r * trans;
    trans += s.getTransVector();
}

void FairGeoTransform::clear()
{
    trans.clear();
    rot.setUnitMatrix();
}

void FairGeoTransform::print()
{
    rot.print();
    trans.print();
}

void FairGeoTransform::invert(void)
{
    rot.invert();
    trans = rot * trans;
    trans *= -1.;
}
