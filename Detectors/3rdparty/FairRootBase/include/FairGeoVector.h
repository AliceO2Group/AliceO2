/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOVECTOR_H
#define FAIRGEOVECTOR_H

#include <Rtypes.h>      // for Double_t, Bool_t, Int_t, etc
#include <TMath.h>       // for pow, floor, sqrt
#include <TMathBase.h>   // for Abs
#include <TObject.h>     // for TObject
#include <cmath>
#include <iostream>   // for operator<<, ostream, etc
#include <stdio.h>    // for printf

class FairGeoVector : public TObject
{
  protected:
    Double_t x;
    Double_t y;
    Double_t z;
    inline void round(Double_t d, Int_t n);

  public:
    FairGeoVector(Double_t dx = 0, Double_t dy = 0, Double_t dz = 0)
        : TObject()
        , x(dx)
        , y(dy)
        , z(dz)
    {}
    FairGeoVector(const FairGeoVector& v)
        : TObject(v)
        , x(v.getX())
        , y(v.getY())
        , z(v.getZ())
    {}
    ~FairGeoVector() {}
    Double_t& X() { return x; }
    Double_t& Y() { return y; }
    Double_t& Z() { return z; }
    Double_t getX() const { return x; }
    Double_t getY() const { return y; }
    Double_t getZ() const { return z; }

    Double_t getValues(Int_t i)
    {
        if (i < 0 || i > 2) {
            std::cout << " -E- Vector index  is 0 1 2 only ! " << std::endl;
        }
        if (i == 0) {
            return x;
        }
        if (i == 1) {
            return y;
        }
        if (i == 2) {
            return z;
        }
        return -1;
    }

    void setXYZ(const Double_t xx, const Double_t yy, const Double_t zz)
    {
        x = xx;
        y = yy;
        z = zz;
    }
    void setX(const Double_t a) { x = a; }
    void setY(const Double_t a) { y = a; }
    void setZ(const Double_t a) { z = a; }
    inline void setVector(const Double_t* a);
    inline void setVector(const Float_t* a);
    inline Double_t operator()(const Int_t i) const;
    inline FairGeoVector operator-() const;
    inline FairGeoVector& operator=(const FairGeoVector& v);
    inline Bool_t operator==(const FairGeoVector& v) const;
    inline Bool_t operator!=(const FairGeoVector& v) const;
    inline Bool_t operator<(const Double_t a);
    inline Bool_t operator<=(const Double_t a);
    inline Bool_t operator>(const Double_t a);
    inline Bool_t operator>=(const Double_t a);
    inline FairGeoVector& operator+=(const Double_t a);
    inline FairGeoVector& operator-=(const Double_t a);
    inline FairGeoVector& operator*=(const Double_t a);
    inline FairGeoVector& operator/=(const Double_t a);
    inline FairGeoVector& operator+=(const FairGeoVector& v);
    inline FairGeoVector& operator-=(const FairGeoVector& v);
    inline FairGeoVector operator+(const FairGeoVector& v) const;
    inline FairGeoVector operator-(const FairGeoVector& v) const;
    inline FairGeoVector& abs();
    inline Double_t scalarProduct(const FairGeoVector& v) const;
    inline FairGeoVector vectorProduct(const FairGeoVector& v) const;
    Double_t length() const { return sqrt(x * x + y * y + z * z); }
    void clear() { x = y = z = 0.; }
    void print() const { printf("%10.3f%10.3f%10.3f\n", x, y, z); }
    inline void round(Int_t n);
    inline friend std::ostream& operator<<(std::ostream& put, const FairGeoVector& v);
    inline friend std::istream& operator>>(std::istream& get, FairGeoVector& v);
    ClassDef(FairGeoVector, 1);   // vector with 3 components
};

// -------------------- inlines ---------------------------

inline void FairGeoVector::setVector(const Double_t* a)
{
    x = a[0];
    y = a[1];
    z = a[2];
}

inline void FairGeoVector::setVector(const Float_t* a)
{
    x = a[0];
    y = a[1];
    z = a[2];
}

inline Double_t FairGeoVector::operator()(const Int_t i) const
{
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            Error("operator()", "bad index");
    }
    return 0;
}

inline FairGeoVector FairGeoVector::operator-() const
{
    FairGeoVector p(-x, -y, -z);
    return p;
}

inline FairGeoVector& FairGeoVector::operator=(const FairGeoVector& v)
{
    x = v.getX();
    y = v.getY();
    z = v.getZ();
    return *this;
}

inline Bool_t FairGeoVector::operator==(const FairGeoVector& v) const
{
    return ((v.getX() != x || v.getY() != y || v.getZ() != z) ? kFALSE : kTRUE);
}

inline Bool_t FairGeoVector::operator!=(const FairGeoVector& v) const
{
    return (v.getX() != x || v.getY() != y || v.getZ() != z) ? kTRUE : kFALSE;
}
/// check with ilse
inline Bool_t FairGeoVector::operator<(const Double_t a) { return (x >= a || y >= a || z >= a) ? kFALSE : kTRUE; }

inline Bool_t FairGeoVector::operator<=(const Double_t a) { return (x > a || y > a || z > a) ? kFALSE : kTRUE; }

inline Bool_t FairGeoVector::operator>(const Double_t a) { return (x <= a || y <= a || z <= a) ? kFALSE : kTRUE; }

inline Bool_t FairGeoVector::operator>=(const Double_t a) { return (x < a || y < a || z < a) ? kFALSE : kTRUE; }

inline FairGeoVector& FairGeoVector::operator+=(const Double_t a)
{
    x += a;
    y += a;
    z += a;
    return *this;
}

inline FairGeoVector& FairGeoVector::operator-=(const Double_t a)
{
    x -= a;
    y -= a;
    z -= a;
    return *this;
}

inline FairGeoVector& FairGeoVector::operator*=(const Double_t a)
{
    x *= a;
    y *= a;
    z *= a;
    return *this;
}

inline FairGeoVector& FairGeoVector::operator/=(const Double_t a)
{
    x /= a;
    y /= a;
    z /= a;
    return *this;
}

inline FairGeoVector& FairGeoVector::operator+=(const FairGeoVector& v)
{
    x += v.getX();
    y += v.getY();
    z += v.getZ();
    return *this;
}

inline FairGeoVector& FairGeoVector::operator-=(const FairGeoVector& v)
{
    x -= v.getX();
    y -= v.getY();
    z -= v.getZ();
    return *this;
}

inline FairGeoVector FairGeoVector::operator+(const FairGeoVector& v) const
{
    FairGeoVector p(*this);
    return p += v;
}

inline FairGeoVector FairGeoVector::operator-(const FairGeoVector& v) const
{
    FairGeoVector p(*this);
    return p -= v;
}

inline FairGeoVector& FairGeoVector::abs()
{
    x = TMath::Abs(x);
    y = TMath::Abs(y);
    z = TMath::Abs(z);
    return *this;
}

inline Double_t FairGeoVector::scalarProduct(const FairGeoVector& v) const
{
    return (x * v.getX() + y * v.getY() + z * v.getZ());
}

inline FairGeoVector FairGeoVector::vectorProduct(const FairGeoVector& v) const
{
    FairGeoVector p(y * v.getZ() - z * v.getY(), z * v.getX() - x * v.getZ(), x * v.getY() - y * v.getX());
    return p;
}

inline void FairGeoVector::round(Double_t d, Int_t n)
{
    // rounds d to a precision with n digits
    if (d > 0) {
        d = floor(d * pow(10., n) + 0.5) / pow(10., n);
    } else {
        d = -floor((-d) * pow(10., n) + 0.5) / pow(10., n);
    }
}

inline void FairGeoVector::round(Int_t n)
{
    // rounds every component to a precision with n digits
    round(x, n);
    round(y, n);
    round(z, n);
}

inline std::ostream& operator<<(std::ostream& put, const FairGeoVector& v)
{
    return put << v(0) << "  " << v(1) << "  " << v(2) << '\n';
}

inline std::istream& operator>>(std::istream& get, FairGeoVector& v)
{
    Double_t x[3];
    get >> x[0] >> x[1] >> x[2];
    v.setVector(x);
    return get;
}

#endif /* !FAIRGEOVECTOR_H */
