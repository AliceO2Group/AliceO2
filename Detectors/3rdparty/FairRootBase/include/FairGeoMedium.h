/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOMEDIUM_H
#define FAIRGEOMEDIUM_H

#include <Rtypes.h>   // for Double_t, Int_t, Bool_t, etc
#include <TNamed.h>   // for TNamed
#include <iosfwd>     // for fstream

/**
 * Class for tracking medium
 * @author ilse koenig
 */
class FairGeoMedium : public TNamed
{
  private:
    Int_t medId;        /** Unique number for the material and the medium */
    Int_t autoflag;     /** Flag for default values of madfld .. minstep*/
    Int_t nComponents;  /** Number of material components */
    Int_t weightFac;    /** Factor for weights (1: relative w., -1: w. by number of atoms)*/
    Double_t* ca;       //[nComponents] /** Atomic weights for the components*/
    Double_t* cz;       //[nComponents] /** Atomic numbers for the components*/
    Double_t* cw;       //[nComponents] /** Weights of the components in a mixture*/
    Double_t density;   /** Density in g cm(**-3)*/
    Double_t radLen;    /** Radiation lenght*/
    Int_t sensFlag;     /** Sensitivity flag*/
    Int_t fldFlag;      /** Field flag*/
    Double_t fld;       /** Maximum field value in kilogauss*/
    Double_t epsil;     /** Boundary crossing precision*/
    Double_t madfld;    /** Maximum angular deviation due to field*/
    Double_t maxstep;   /** Maximum step permitted*/
    Double_t maxde;     /** Maximum fractional energy loss*/
    Double_t minstep;   /** Minimum value for step*/
    Int_t npckov;       /** Number of optical parameters for Cerenkov*/
    Double_t* ppckov;   //[npckov]  /** Photon momentum*/
    Double_t* absco;    //[npckov] /** Absoption length*/
    Double_t* effic;    //[npckov] /** Detection efficiency*/
    Double_t* rindex;   //[npckov] /** Refraction index*/
  public:
    FairGeoMedium(const char* name = "");
    FairGeoMedium& operator=(const FairGeoMedium&) { return *this; }
    ~FairGeoMedium();
    void setMediumIndex(Int_t i) { medId = i; }
    Int_t getMediumIndex() { return medId; }
    void setAutoFlag(Int_t i) { autoflag = i; }
    Int_t getAutoFlag() { return autoflag; }
    void setNComponents(Int_t);
    Int_t getNComponents() { return nComponents; }
    Bool_t setComponent(Int_t, Double_t, Double_t, Double_t w = 1.);
    void getComponent(Int_t, Double_t*);
    Int_t getWeightFac() { return weightFac; }
    void setDensity(Double_t d) { density = d; }
    Double_t getDensity() { return density; }
    void setRadiationLength(Double_t l) { radLen = l; }
    Double_t getRadiationLength() { return radLen; }
    void setNpckov(Int_t);
    Int_t getNpckov() { return npckov; }
    Bool_t setCerenkovPar(Int_t, Double_t, Double_t, Double_t, Double_t);
    void getCerenkovPar(Int_t, Double_t*);
    void setMediumPar(Int_t,
                      Int_t,
                      Double_t,
                      Double_t,
                      Double_t maxDeviation = -1.,
                      Double_t maxStep = -1.,
                      Double_t maxDE = -1.,
                      Double_t minStepDouble_t = -1.);
    void getMediumPar(Double_t*);
    inline Bool_t isSensitive();
    Int_t getSensitivityFlag() { return sensFlag; }
    Int_t getFieldFlag() { return fldFlag; }
    Double_t getField() { return fld; }
    Double_t getEpsil() { return epsil; }
    void read(std::fstream&, Int_t autoflag);
    void print();
    void write(std::fstream&);
    Bool_t calcRadiationLength();

  private:
    FairGeoMedium(const FairGeoMedium&);
    // TODO: correct copy constructor for FairGeoMedium
    //    FairGeoMedium& operator=(const FairGeoMedium&);

    ClassDef(FairGeoMedium, 1);   //
};

inline Bool_t FairGeoMedium::isSensitive()
{
    if (sensFlag) {
        return kTRUE;
    } else {
        return kFALSE;
    }
}

#endif /* !FAIRGEOMEDIUM_H */
