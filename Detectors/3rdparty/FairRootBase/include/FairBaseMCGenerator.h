/*
 * FairBaseMCGenerator.h
 *
 *  Created on: 16 mar 2020
 *      Author: Daniel Wielanek
 *		E-mail: daniel.wielanek@gmail.com
 *		Warsaw University of Technology, Faculty of Physics
 */
#ifndef FAIR_BASEMCGENERATOR_H_
#define FAIR_BASEMCGENERATOR_H_

#include "FairGenerator.h"

/**
 * abstract class for generators that use "macro input" to generate partcles
 */
class FairBaseMCGenerator : public FairGenerator
{
  public:
    enum eVertexSmear
    {
        kBox,
        kGauss,
        kExp
    };
    FairBaseMCGenerator();
    /**
     * set PDG code of generated particles
     * @param pdg
     */
    void SetPDGType(Int_t pdg);
    /**
     * set vertex position and algorithm of vertex generation
     * @param vx, vy, vz Vertex coordinates [cm]
     * @param evx, evy, evz Vertex dispersion [cm]
     * @param sm distribution of vertex, gaussian or box
     */
    void SetVertex(Double_t vx,
                   Double_t vy,
                   Double_t vz,
                   Double_t evx = 0,
                   Double_t evy = 0,
                   Double_t evz = 0,
                   eVertexSmear sm = kBox);
    /**
     * set number of particles per event
     * @param mult
     */
    void SetMultiplicity(Int_t mult) { fMult = mult; };
    virtual Bool_t Init();
    virtual ~FairBaseMCGenerator();

  protected:
    FairBaseMCGenerator(const FairBaseMCGenerator &other) = default;
    FairBaseMCGenerator &operator=(const FairBaseMCGenerator &rhs) = default;
    Int_t GetPDGType() const { return fPDGType; };
    Int_t GetMultiplicity() const { return fMult; };
    Double_t GetPDGMass() const { return fPDGMass; };
    /**
     * generate event parameters like current event
     */
    virtual void GenerateEventParameters();
    Double_t GetVxAv() const { return fVx; };
    Double_t GetVyAv() const { return fVy; };
    Double_t GetVzAv() const { return fVz; };
    Double_t GetVxRms() const { return fVex; }
    Double_t GetVyRms() const { return fVey; }
    Double_t GetVzRms() const { return fVez; }
    Double_t fX, fY, fZ;   // current vertex coordinates [cm]
  private:
    Int_t fPDGType;   // Particle type (PDG encoding)
    Int_t fMult;      // Multiplicity;
    eVertexSmear fVertexSmear;
    Double_t fPDGMass;
    Double_t fVx, fVy, fVz;
    Double_t fVex, fVey, fVez;
    ClassDef(FairBaseMCGenerator, 1);
};

#endif /* FAIR_BASEMCGENERATOR_H_ */
