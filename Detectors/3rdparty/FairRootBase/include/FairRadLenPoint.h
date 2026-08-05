/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairRadLenPoint header file               -----
// -----                  Created 14/01/08  by M. Al-Turany            -----
// -------------------------------------------------------------------------

/**  FairRadLenPoint.h
 *@author M. Al-Turany
 *
 */

#ifndef FAIRRADLENPOINT_H
#define FAIRRADLENPOINT_H

#include "FairMCPoint.h"   // for FairMCPoint

#include <Rtypes.h>     // for Double_t, Float_t, etc
#include <TVector3.h>   // for TVector3

class FairRadLenPoint : public FairMCPoint
{
  public:
    /** Default constructor **/
    FairRadLenPoint();

    FairRadLenPoint(Int_t trackID,
                    Int_t detID,
                    TVector3 pos,
                    TVector3 mom,
                    Double_t tof,
                    Double_t length,
                    Double_t eLoss,
                    TVector3 posOut,
                    TVector3 momOut,
                    Float_t fA,
                    Float_t fZ,
                    Float_t fDensity,
                    Float_t fRadLen);

    /** Destructor **/
    virtual ~FairRadLenPoint();

    /** Accessors **/
    Float_t GetA() { return fA; }
    Float_t GetZm() { return fZmat; }
    Float_t GetRadLength() { return fRadLen; }
    Float_t GetDensity() { return fDensity; }

    /** Modifiers **/
    void SetA(Double_t A) { fA = A; }
    void SetZm(Double_t Z) { fZmat = Z; }
    void SetRadLength(Double_t length) { fRadLen = length; }
    void SetDensity(Double_t Density) { fDensity = Density; }

    /** Output to screen **/
    virtual void Print(const Option_t* opt) const;

    Double_t GetXOut() const { return fXOut; };
    Double_t GetYOut() const { return fYOut; };
    Double_t GetZOut() const { return fZOut; };

    Double_t GetPxOut() const { return fPxOut; }
    Double_t GetPyOut() const { return fPyOut; }
    Double_t GetPzOut() const { return fPzOut; }

    TVector3 GetPosition() const { return TVector3(fX, fY, fZ); }
    TVector3 GetPositionOut() const { return TVector3(fXOut, fYOut, fZOut); }

    void PositionOut(TVector3& pos) const { pos.SetXYZ(fXOut, fYOut, fZOut); };
    void MomentumOut(TVector3& mom) const { mom.SetXYZ(fPxOut, fPyOut, fPzOut); }

  protected:
    Float_t fA;         // A of material
    Float_t fZmat;      // Z of material
    Float_t fDensity;   // density of material
    Float_t fRadLen;    // radiation length

    Double_t fXOut, fYOut, fZOut;
    Double_t fPxOut, fPyOut, fPzOut;

    ClassDef(FairRadLenPoint, 1);
};

#endif
