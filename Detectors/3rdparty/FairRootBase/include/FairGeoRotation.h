/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOROTATION_H
#define FAIRGEOROTATION_H

#include "FairGeoVector.h"   // for FairGeoVector

#include <Rtypes.h>    // for Double_t, Int_t, Bool_t, etc
#include <TObject.h>   // for TObject
#include <iostream>    // for operator<<, cout, ostream, etc

class TRotMatrix;

/**
 * rotation matrix
 * @author Ilse koenig
 */
class FairGeoRotation : public TObject
{
  protected:
    Double_t rot[9];

  public:
    inline FairGeoRotation();
    inline FairGeoRotation(const FairGeoRotation&);
    inline FairGeoRotation(const Double_t*);
    FairGeoRotation(const Double_t, const Double_t, const Double_t);
    ~FairGeoRotation() {}
    inline void setMatrix(const Double_t*);
    inline void setMatrix(const Float_t*);
    void setEulerAngles(const Double_t, const Double_t, const Double_t);
    inline void setElement(const Double_t, const Int_t);
    inline Double_t operator()(Int_t) const;
    inline FairGeoRotation& operator=(const FairGeoRotation&);
    inline Bool_t operator==(const FairGeoRotation&);
    inline Bool_t operator!=(const FairGeoRotation&);
    inline FairGeoVector operator*(const FairGeoVector&)const;
    inline FairGeoRotation operator*(const FairGeoRotation&)const;
    inline FairGeoRotation& operator*=(const FairGeoRotation&);
    inline FairGeoRotation& transform(const FairGeoRotation&);
    inline Bool_t isUnitMatrix();
    inline FairGeoRotation inverse() const;
    inline FairGeoRotation& invert();
    inline Double_t determinant() const;
    Double_t diff2(const FairGeoRotation&) const;
    inline Double_t getElement(Int_t i, Int_t j) const;
    inline void setUnitMatrix();
    inline void setZero();
    inline void print() const;
    TRotMatrix* createTRotMatrix(const Text_t* name = "", const Text_t* title = "");

    ClassDef(FairGeoRotation, 1);   //
};

// -------------------- inlines ---------------------------

inline FairGeoRotation::FairGeoRotation()
    : TObject()
{
    rot[0] = rot[4] = rot[8] = 1.;
    rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.;
}

inline Double_t FairGeoRotation::operator()(Int_t i) const
{
    if (i >= 0 && i < 9) {
        return rot[i];
    }
    Error("operator()", "bad index");
    return 0;
}

inline FairGeoRotation::FairGeoRotation(const FairGeoRotation& r)
    : TObject(r)
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = r(i);
    }
}

inline FairGeoRotation::FairGeoRotation(const Double_t* a)
    : TObject()
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = a[i];
    }
}

inline void FairGeoRotation::setMatrix(const Double_t* a)
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = a[i];
    }
}

inline void FairGeoRotation::setMatrix(const Float_t* a)
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = a[i];
    }
}

inline void FairGeoRotation::setElement(const Double_t a, const Int_t i)
{
    if (i < 9) {
        rot[i] = a;
    }
}

inline Double_t FairGeoRotation::getElement(Int_t i, Int_t j) const { return rot[i * 3 + j]; }

inline FairGeoRotation& FairGeoRotation::operator=(const FairGeoRotation& r)
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = r(i);
    }
    return *this;
}

inline Bool_t FairGeoRotation::operator==(const FairGeoRotation& r)
{
    Int_t i = 0;
    while (i < 9) {
        if (rot[i] != r(i)) {
            return kFALSE;
        }
        i++;
    }
    return kTRUE;
}

inline Bool_t FairGeoRotation::operator!=(const FairGeoRotation& r)
{
    Int_t i = 0;
    while (i < 9) {
        if (rot[i] != r(i)) {
            return kTRUE;
        }
        i++;
    }
    return kFALSE;
}

inline FairGeoVector FairGeoRotation::operator*(const FairGeoVector& v) const
{
    return FairGeoVector(rot[0] * v(0) + rot[1] * v(1) + rot[2] * v(2),
                         rot[3] * v(0) + rot[4] * v(1) + rot[5] * v(2),
                         rot[6] * v(0) + rot[7] * v(1) + rot[8] * v(2));
}

inline FairGeoRotation FairGeoRotation::operator*(const FairGeoRotation& r) const
{
    Double_t a[9];
    for (Int_t kk = 0; kk < 9; kk++) {
        a[kk] = 0;
    }
    for (Int_t i = 0; i < 3; i++) {
        for (Int_t j = 0; j < 3; j++) {
            Int_t n = 3 * i + j;
            for (Int_t k = 0; k < 3; k++) {
                a[n] += rot[3 * i + k] * r(3 * k + j);
            }
        }
    }
    return FairGeoRotation(&a[0]);
}

inline FairGeoRotation& FairGeoRotation::operator*=(const FairGeoRotation& r) { return *this = operator*(r); }

inline FairGeoRotation& FairGeoRotation::transform(const FairGeoRotation& r) { return *this = r * (*this); }

inline Bool_t FairGeoRotation::isUnitMatrix()
{
    return (rot[0] == 1. && rot[1] == 0. && rot[2] == 0. && rot[3] == 0. && rot[4] == 1. && rot[5] == 0. && rot[6] == 0.
            && rot[7] == 0. && rot[8] == 1.)
               ? kTRUE
               : kFALSE;
}

inline FairGeoRotation FairGeoRotation::inverse() const
{
    Double_t a[9];
    for (Int_t i = 0; i < 3; i++) {
        for (Int_t j = 0; j < 3; j++) {
            a[j + 3 * i] = rot[i + 3 * j];
        }
    }
    return FairGeoRotation(a);
}

inline FairGeoRotation& FairGeoRotation::invert() { return *this = inverse(); }

inline Double_t FairGeoRotation::determinant() const
{
    return rot[0] * (rot[4] * rot[8] - rot[7] * rot[5]) - rot[3] * (rot[1] * rot[8] - rot[7] * rot[2])
           + rot[6] * (rot[1] * rot[5] - rot[4] * rot[2]);
}

inline void FairGeoRotation::setUnitMatrix()
{
    rot[0] = rot[4] = rot[8] = 1.;
    rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.;
}

inline void FairGeoRotation::setZero()
{
    for (Int_t i = 0; i < 9; i++) {
        rot[i] = 0.;
    }
}

inline void FairGeoRotation::print() const
{
    for (Int_t i = 0; i < 9; i++) {
        std::cout << rot[i] << "  ";
    }
    std::cout << '\n';
}

#endif /* !FAIRGEOROTATION_H */
