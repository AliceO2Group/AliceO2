/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                     FairRadMapPoint header file               -----
// -------------------------------------------------------------------------

/**  FairRadMapPoint.h
 *@author O.N. Hartmann
 *
 */

#ifndef FAIRRADMAPPOINT_H
#define FAIRRADMAPPOINT_H

#include "FairMCPoint.h"   // for FairMCPoint

#include <Rtypes.h>     // for Double_t, Float_t, Int_t, etc
#include <TVector3.h>   // for TVector3

class FairRadMapPoint : public FairMCPoint
{
  public:
    /** Default constructor **/
    FairRadMapPoint();

    FairRadMapPoint(Int_t trackID,
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
                    Double_t fVolMass,
                    Double_t fStep,
                    Double_t fDose,
                    Double_t fDoseSL,
                    Int_t fPdg);

    /** Destructor **/
    virtual ~FairRadMapPoint();

    /** Accessors **/
    Float_t GetA() { return fA; }
    Float_t GetZm() { return fZmat; }
    Float_t GetMass() { return fVolMass; }
    Float_t GetDensity() { return fDensity; }

    /** Modifiers **/
    void SetA(Double_t A) { fA = A; }
    void SetZm(Double_t Z) { fZmat = Z; }
    void SetMass(Double_t vmass) { fVolMass = vmass; }
    void SetDensity(Double_t Density) { fDensity = Density; }

    /** Output to screen **/
    virtual void Print(const Option_t* opt) const;

    Int_t GetPdg() const { return fPdg; };

    Double_t GetXOut() const { return fXOut; }
    Double_t GetYOut() const { return fYOut; }
    Double_t GetZOut() const { return fZOut; }

    Double_t GetPxOut() const { return fPxOut; }
    Double_t GetPyOut() const { return fPyOut; }
    Double_t GetPzOut() const { return fPzOut; }

    Double_t GetDose() const { return fDose; }
    Double_t GetDoseSL() const { return fDoseSL; }

    TVector3 GetPosition() const { return TVector3(fX, fY, fZ); }
    TVector3 GetPositionOut() const { return TVector3(fXOut, fYOut, fZOut); }

    void PositionOut(TVector3& pos) const { pos.SetXYZ(fXOut, fYOut, fZOut); };
    void MomentumOut(TVector3& mom) const { mom.SetXYZ(fPxOut, fPyOut, fPzOut); }

  protected:
    // Int_t detID; // detector ID or volume ID
    Int_t fPdg;          // track PID
    Float_t fA;          // A of material
    Float_t fZmat;       // Z of material
    Float_t fDensity;    // density of material
    Double_t fVolMass;   // radiation length

    Double_t fStep, fDose, fDoseSL;
    Double_t fXOut, fYOut, fZOut;
    Double_t fPxOut, fPyOut, fPzOut;

    ClassDef(FairRadMapPoint, 3);
};

#endif
