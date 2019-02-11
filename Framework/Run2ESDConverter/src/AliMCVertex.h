#ifndef AliMCVertex_H
#define AliMCVertex_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//     Simple MC vertex class
//     To be used with Kinematics MC data
//     Inherits from AliVVertex
//-------------------------------------------------------------------------

#include "AliVVertex.h"

class AliMCVertex : public AliVVertex {
    public:
  AliMCVertex() {fPosition[0] = 0.; fPosition[1] = 0.; fPosition[2] = 0.;}
    AliMCVertex(Double_t xv, Double_t yv, Double_t zv)
	{fPosition[0] = xv; fPosition[1] = yv; fPosition[2] = zv;}
    virtual ~AliMCVertex() {;}
    virtual void     GetXYZ(Double_t position[3]) const
	{for(Int_t i = 0; i < 3; i++) position[i] = fPosition[i];}
    virtual Double_t GetX() const {return fPosition[0];}
    virtual Double_t GetY() const {return fPosition[1];}
    virtual Double_t GetZ() const {return fPosition[2];}
    virtual void     GetCovarianceMatrix(Double_t covmatrix[6]) const
	{for(Int_t i = 0; i < 6; i++) covmatrix[i] = 0.;}
    virtual Double_t GetChi2perNDF() const {return -1.;}
    virtual Double_t GetChi2()       const {return -1.;}
    virtual Int_t    GetNDF()        const {return -1 ;}

    virtual void     SetPosition(Double_t xv, Double_t yv, Double_t zv)
	{fPosition[0] = xv; fPosition[1] = yv; fPosition[2] = zv;}
    virtual Int_t    GetNContributors() const {return -1;}
    virtual void     PrintIndices() const {;}
    virtual void     Print(Option_t* option = "") const;
    virtual void     Clear(Option_t* option) {TNamed::Clear(option);}
 private:
    Double32_t      fPosition[3]; // vertex position
    ClassDef(AliMCVertex, 1);
};

#endif
