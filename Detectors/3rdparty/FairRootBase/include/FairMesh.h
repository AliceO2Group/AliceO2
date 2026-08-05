/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                       FairMesh header file                    -----
// -----          original author                  D.Bertini           -----
// -----          adapted april 2010               O.Hartmann          -----
// -------------------------------------------------------------------------

#ifndef FAIRMESH_H
#define FAIRMESH_H

#include <Rtypes.h>    // for Double_t, Int_t, etc
#include <TH2.h>       // for TH2D
#include <TObject.h>   // for TObject
#include <TString.h>   // for TString

/**
 * Abstract base class for Mesh Objects.
 **@author D.Bertini <d.bertini@gsi.de>
 */
class FairMesh : public TObject
{
  public:
    /** Default constructor **/
    FairMesh();
    FairMesh(const char*);

    /** Destructor **/
    virtual ~FairMesh();

    /** Accessors **/
    Double_t GetXmin() const { return fXmin; };
    Double_t GetYmin() const { return fYmin; };
    Double_t GetZmin() const { return fZmin; };

    Double_t GetXmax() const { return fXmax; };
    Double_t GetYmax() const { return fYmax; };
    Double_t GetZmax() const { return fZmax; };

    /** Modifiers **/
    void SetX(Double_t xmin, Double_t xmax, Int_t nbin)
    {
        fXmin = xmin;
        fXmax = xmax;
        NXbin = nbin;
    }
    void SetY(Double_t ymin, Double_t ymax, Int_t nbin)
    {
        fYmin = ymin;
        fYmax = ymax;
        NYbin = nbin;
    }
    void SetZ(Double_t zmin, Double_t zmax, Int_t nbin)
    {
        fZmin = zmin;
        fZmax = zmax;
        NZbin = nbin;
    }

    TH2D* GetMeshTid() { return fMeshTid; }
    TH2D* GetMeshFlu() { return fMeshFlu; }
    TH2D* GetMeshSEU() { return fMeshSEU; }
    /*** Output to screen */
    // virtual void Print(const Option_t* opt = 0) const {;}

    Double_t GetDiag() { return fDiag; }
    Double_t GetBinVolume() { return fBinVolume; }

    void fillTID(Double_t x, Double_t y, Double_t we) { fMeshTid->Fill(x, y, we); }
    void fillFluence(Double_t x, Double_t y, Double_t we) { fMeshFlu->Fill(x, y, we); }
    void fillSEU(Double_t x, Double_t y, Double_t we) { fMeshSEU->Fill(x, y, we); }
    void Scale(Double_t fac)
    {
        fMeshTid->Scale(fac);
        fMeshFlu->Scale(fac);
        fMeshSEU->Scale(fac);
    }

    void calculate();

    void print();

  protected:
    Double_t fXmin, fYmin, fZmin;
    Double_t fXmax, fYmax, fZmax;
    Int_t NXbin, NYbin, NZbin;
    Double_t fBinVolume;
    Double_t fDiag;
    TH2D* fMeshTid;   // !mesh
    TH2D* fMeshFlu;   // !mesh
    TH2D* fMeshSEU;   // !mesh
    TString fhname;   // !mesh

  private:
    FairMesh(const FairMesh&);
    FairMesh& operator=(const FairMesh&);

    ClassDef(FairMesh, 1);
};

#endif
