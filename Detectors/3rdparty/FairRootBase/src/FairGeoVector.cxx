/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 16/06/99

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoVector
//
// This class defines a vector with 3 Double_t components.
// Instantiations of this class are e.g geometry points and
// translation vectors in the geometry transformations.
//
// All functions and operators are defined inline.
//
// Constructors:
//   FairGeoVector(Double_t dx=0,Double_t dy=0,Double_t dz=0)
//   FairGeoVector(const FairGeoVector& v)
//
// Access to the components:
//   void setVector(const Double_t* a);
//   void setX(const Double_t a)
//   void setY(const Double_t a)
//   void setZ(const Double_t a)
//   Double_t getX() const
//   Double_t getY() const
//   Double_t getZ() const
//   Double_t operator() (const Int_t i) const;
//
// Check for the size of all components:
//   Bool_t operator < (const Double_t a)
//   Bool_t operator <= (const Double_t a)
//   Bool_t operator > (const Double_t a)
//   Bool_t operator >= (const Double_t a)
//
// Check for equality/inequality;
//   Bool_t operator == (const FairGeoVector& v) const
//   Bool_t operator != (const FairGeoVector& v) const
//
// Addition/Substraction of a vector;
//   FairGeoVector& operator += (const Double_t a)
//   FairGeoVector& operator -= (const Double_t a)
//
// Addition/Substraction of two vectors;
//   FairGeoVector operator + (const FairGeoVector& v) const
//   FairGeoVector operator - (const FairGeoVector& v) const
//
// Multiplication/Division of each components with/by a factor
//   FairGeoVector& operator *= (const Double_t a)
//   FairGeoVector& operator /= (const Double_t a)
//
//
// FairGeoVector operator - () const
//      changes the sign of all components
//
// FairGeoVector& operator = (const FairGeoVector& v)
//      assignment
//
// FairGeoVector& abs()
//      builds absolute value of each components
//
// Double_t scalarProduct(const FairGeoVector& v) const;
//      returns the scalar product
//
// FairGeoVector vectorProduct(const FairGeoVector& v) const;
//      returns the vector product
//
// Double_t length() const {return sqrt(x*x+y*y+z*z);}
//      returns the length
//
//  void clear()
//      sets all components to 0.0
//
// void print()
//      prints the components
//
// void round(Int_t n)
//      rounds all components to a precision with n digits
//
// friend ostream& operator << (ostream& put,const FairGeoVector& v);
//      output to stream
//
// friend istream& operator >> (istream& get,FairGeoVector& v);
//      input from stream
//
///////////////////////////////////////////////////////////////////////////////

#include "FairGeoVector.h"

ClassImp(FairGeoVector);
