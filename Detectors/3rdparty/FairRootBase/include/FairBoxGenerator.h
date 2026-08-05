/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                FairBoxGenerator header file                    -----
// -----          Created 09/09/04  by Yu.Kharlov                      -----
// -------------------------------------------------------------------------

/**  FairBoxGenerator.h
 *@author Yuri Kharlov <Yuri.Kharlov@cern.ch>
 *
 * The FairBoxGenerator generates particles within given range
 * over pt, phi, eta, y or theta a fixed multiplicity per event.
 * Derived from FairGenerator.
 **/

/* $Id: FairBoxGenerator.h,v 1.3 2006/07/14 11:23:57 kharlov Exp $ */

/* History of cvs commits:
 *
 * $Log: FairBoxGenerator.h,v $
 * Revision 1.3  2006/07/14 11:23:57  kharlov
 * Add protection for simultaneously set ranges; split vertex and kinematics ranges
 *
 * Revision 1.2  2006/03/29 16:25:50  kharlov
 * New functionality added
 *
 */

#ifndef FAIR_BOXGENERATOR_H
#define FAIR_BOXGENERATOR_H

#include "FairBaseMCGenerator.h"
#include "FairGenerator.h"   // for FairGenerator

#include <Rtypes.h>   // for Double32_t, Bool_t, kTRUE, etc

class FairPrimaryGenerator;

class FairBoxGenerator : public FairBaseMCGenerator
{
  public:
    /** Default constructor. **/
    FairBoxGenerator();

    /** Constructor with PDG-ID, multiplicity
     **@param pdgid Particle type (PDG encoding)
     **@param mult  Multiplicity (default is 1)
     **/
    FairBoxGenerator(Int_t pdgid, Int_t mult = 1);

    /** Destructor **/
    virtual ~FairBoxGenerator() {}

    /** Modifiers **/

    void SetPRange(Double32_t pmin = 0, Double32_t pmax = 10)
    {
        fPMin = pmin;
        fPMax = pmax;
        fPRangeIsSet = kTRUE;
    }

    void SetPtRange(Double32_t ptmin = 0, Double32_t ptmax = 10)
    {
        fPtMin = ptmin;
        fPtMax = ptmax;
        fPtRangeIsSet = kTRUE;
    }

    void SetEkinRange(Double32_t kmin = 0, Double32_t kmax = 10)
    {
        fEkinMin = kmin;
        fEkinMax = kmax;
        fEkinRangeIsSet = kTRUE;
    }

    void SetPhiRange(Double32_t phimin = 0, Double32_t phimax = 360)
    {
        fPhiMin = phimin;
        fPhiMax = phimax;
    }

    void SetEtaRange(Double32_t etamin = -5, Double32_t etamax = 7)
    {
        fEtaMin = etamin;
        fEtaMax = etamax;
        fEtaRangeIsSet = kTRUE;
    }

    void SetYRange(Double32_t ymin = -5, Double32_t ymax = 7)
    {
        fYMin = ymin;
        fYMax = ymax;
        fYRangeIsSet = kTRUE;
    }

    void SetThetaRange(Double32_t thetamin = 0, Double32_t thetamax = 90)
    {
        fThetaMin = thetamin;
        fThetaMax = thetamax;
        fThetaRangeIsSet = kTRUE;
    }

    void SetCosTheta() { fCosThetaIsSet = kTRUE; }

    void SetXYZ(Double32_t x = 0, Double32_t y = 0, Double32_t z = 0);

    void SetBoxXYZ(Double32_t x1 = 0, Double32_t y1 = 0, Double32_t x2 = 0, Double32_t y2 = 0, Double32_t z = 0);

    /**
     * not used, for backward compatibility, please user FairLogger to set debug mode
     * @param
     */
    void SetDebug(Bool_t /*debug*/) {}
    /** Initializer **/
    Bool_t Init();

    /** Creates an event with given type and multiplicity.
     **@param primGen  pointer to the FairPrimaryGenerator
     **/
    virtual Bool_t ReadEvent(FairPrimaryGenerator* primGen);

    /** Clone this object (used in MT mode only) */
    virtual FairGenerator* CloneGenerator() const;

  protected:
    /** Copy constructor. **/
    FairBoxGenerator(const FairBoxGenerator&) = default;
    FairBoxGenerator& operator=(const FairBoxGenerator&) = default;

  private:
    Double32_t fPtMin, fPtMax;         // Transverse momentum range [GeV]
    Double32_t fPhiMin, fPhiMax;       // Azimuth angle range [degree]
    Double32_t fEtaMin, fEtaMax;       // Pseudorapidity range in lab system
    Double32_t fYMin, fYMax;           // Rapidity range in lab system
    Double32_t fPMin, fPMax;           // Momentum range in lab system
    Double32_t fThetaMin, fThetaMax;   // Polar angle range in lab system [degree]
    Double32_t fEkinMin, fEkinMax;     // Kinetic Energy range in lab system [GeV]

    Bool_t fEtaRangeIsSet;     // True if eta range is set
    Bool_t fYRangeIsSet;       // True if rapidity range is set
    Bool_t fThetaRangeIsSet;   // True if theta range is set
    Bool_t fCosThetaIsSet;     // True if uniform distribution in
    // cos(theta) is set (default -> not set)
    Bool_t fPtRangeIsSet;     // True if transverse momentum range is set
    Bool_t fPRangeIsSet;      // True if abs.momentum range is set
    Bool_t fEkinRangeIsSet;   // True if kinetic energy range is set

    ClassDef(FairBoxGenerator, 5);
};

#endif
