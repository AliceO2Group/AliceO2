#ifndef ALIVERTEX_H
#define ALIVERTEX_H
/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


//-------------------------------------------------------
//                    Base Vertex Class
//   Used for secondary vertices and as a base class for primary vertices
//   Origin: F. Prino, Torino, prino@to.infn.it
//-------------------------------------------------------

#include <TString.h>
#include "AliVVertex.h"

class AliVertex : public AliVVertex {
 
 public:
 
  AliVertex();
  AliVertex(const Double_t position[3],Double_t dispersion,
		Int_t nContributors);
  AliVertex(const AliVertex &source);
  AliVertex &operator=(const AliVertex &source);
  virtual ~AliVertex();

  virtual void   Clear(Option_t *option="");
  virtual void   SetXYZ(Double_t pos[3]) 
                   {for(Int_t j=0; j<3; j++) fPosition[j]=pos[j];}
  virtual void   SetXv(Double_t xVert) {fPosition[0]=xVert; }
  virtual void   SetYv(Double_t yVert) {fPosition[1]=yVert; }
  virtual void   SetZv(Double_t zVert) {fPosition[2]=zVert; }
  virtual void   SetDispersion(Double_t disp) { fSigma=disp; }
  virtual void   SetNContributors(Int_t nContr) {fNContributors=nContr; }

  virtual void     GetXYZ(Double_t position[3]) const;

  virtual Double_t GetX()  const { return fPosition[0]; }
  virtual Double_t GetY()  const { return fPosition[1]; }
  virtual Double_t GetZ()  const { return fPosition[2]; }
  virtual Double_t GetDispersion() const { return fSigma; }
  virtual Int_t    GetNContributors() const { return fNContributors; }
  virtual Int_t    GetNIndices() const { return fNIndices; }
  virtual Bool_t   GetStatus() const {
    TString title = GetTitle();
    if(fNContributors>0 || (title.Contains("cosmics") && !title.Contains("failed"))) return 1;
    if(title.Contains("smearMC")) return 1;
    return 0;
  }
  virtual Bool_t IsFromVertexer3D() const {
    TString title = GetTitle();  
    if(title.Contains("vertexer: 3D")) return kTRUE;
    else return kFALSE;
  }
  virtual Bool_t IsFromVertexerZ() const {
    TString title = GetTitle();  
    if(title.Contains("vertexer: Z")) return kTRUE;
    else return kFALSE;
  }

  virtual void     Print(Option_t* option = "") const;
  virtual void     SetIndices(Int_t nindices,UShort_t *indices); 
  virtual UShort_t *GetIndices() const { return fIndices; }
  virtual Bool_t   UsesTrack(Int_t index) const;
  virtual Bool_t   SubstituteTrack(Int_t indexOld, Int_t indexNew);
  virtual void     PrintIndices() const {
    if(fNIndices>0) {
      for(Int_t i=0;i<fNIndices;i++) { printf("AliVertex uses track %d\n",fIndices[i]); }
      }
    return;
  }

  virtual void     GetCovarianceMatrix(Double_t covmatrix[6]) const;
  virtual void     SetCovarianceMatrix(const Double_t *) {}
  
  virtual Double_t GetChi2perNDF() const {return -999.;}
  virtual Double_t GetChi2() const {return -999.;}
  virtual void     SetChi2(Double_t ) {}
  virtual Int_t    GetNDF() const {return -999;}

 protected:

  Double32_t fPosition[3];    // vertex position
  Double32_t fSigma;          // track dispersion around found vertex
  Int_t    fNContributors;  // # of tracklets/tracks used for the estimate 
  Int_t    fNIndices;       // # of indices 
  UShort_t *fIndices;       //[fNIndices] indices of tracks used for vertex


  ClassDef(AliVertex,4)  // Class for Primary Vertex
};

#endif
