/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                       FairIon header file                      -----
// -----          Created 27/08/04  by V. Friese / D.Bertini           -----
// -------------------------------------------------------------------------

/** FairIon.h
 *@author V.Friese <v.friese@gsi.de>
 *@author D.Bertini <d.bertini@gsi.de>
 **
 ** A class for the user definition of an ion. It will be instantiated
 ** from the constructor of the FairIonGenerator.
 **/

#ifndef FAIRION_H
#define FAIRION_H

#include <Rtypes.h>   // for Int_t, Double_t, etc
#include <TNamed.h>   // for TNamed

class FairLogger;

class FairIon : public TNamed
{
  public:
    /** Default constructor **/
    FairIon();

    /** Standard constructor
     *@param name  name
     *@param z     atomic number
     *@param a     atomic mass
     *@param q     electric charge
     *@param e     excitation energy
     *@param m     mass [GeV]
     ** If mass is not given, it will be set to a times the proton mass.
     **/

    FairIon(const char* name, Int_t z, Int_t a, Int_t q, Double_t e = 0., Double_t m = 0.);
    void SetParams(const char* name, Int_t z, Int_t a, Int_t q, Double_t e = 0., Double_t m = 0.)
    {
        SetName(name);
        fZ = z;
        fA = a;
        fQ = q;
        fExcEnergy = e;
        fMass = m;
    }
    /** Destructor **/
    virtual ~FairIon();

    /** Accessors **/
    /**
     * Return the atomic number
     */
    Int_t GetZ() const { return fZ; }
    /**
     * Return the atomic mass
     */
    Int_t GetA() const { return fA; }
    /**
     * Return the charge
     */
    Int_t GetQ() const { return fQ; }
    /**
     * Return the  excitation energy
     */
    Double_t GetExcEnergy() const { return fExcEnergy; }
    /**
     * Return the mass in GeV
     */
    Double_t GetMass() const { return fMass; }

    /** Modifiers **/
    /**
     * Set the  excitation energy
     */
    void SetExcEnergy(Double_t eExc) { fExcEnergy = eExc; }
    /**
     * Set the atomic mass, use SetMass to set the mass of the ion
     */
    void SetA(Int_t a) { fA = a; }
    /**
     * Set the mass in GeV
     */
    void SetMass(Double_t mass) { fMass = mass; }

  private:
    static Int_t fgNIon;         //! /// Number of ions instantiated. One per generator.
    Int_t fZ;                    /// Atomic number
    Int_t fA;                    /// Atomic mass
    Int_t fQ;                    /// Electric charge
    Double_t fExcEnergy;         /// Excitation energy [GeV]
    Double_t fMass;              /// Mass [GeV]
    FairLogger* fLogger;         //! /// FairLogger

    FairIon(const FairIon&);
    FairIon& operator=(const FairIon&);

    ClassDef(FairIon, 2);
};

#endif
