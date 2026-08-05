/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairRadLenManager source file             -----
// -----                  Created 14/01/08  by M. Al-Turany            -----
// -------------------------------------------------------------------------
#ifndef FAIRRADLENMANAGER_H
#define FAIRRADLENMANAGER_H 1

#include <Rtypes.h>           // for Float_t, Double_t, Int_t, etc
#include <TLorentzVector.h>   // for TLorentzVector

class TClonesArray;

/**
 * @class FairRadLenManager
 */

class FairRadLenManager
{
  public:
    /**
     * Default constructor.
     * Creates the singleton object of FairRadLenManager class.
     * The pointer to this object can be reached via FairRadLenManager::Instance().
     */
    FairRadLenManager();

    /**
     * Destructor.
     */
    virtual ~FairRadLenManager();

    /**
     * Class definition.
     */
    ClassDef(FairRadLenManager, 1);

  private:
    FairRadLenManager(const FairRadLenManager&);
    FairRadLenManager& operator=(const FairRadLenManager&);

    static FairRadLenManager* fgInstance;

    /**
     * collection of point
     */
    TClonesArray* fPointCollection;
    /**track index */
    Int_t fTrackID;   //!
    /**volume id */
    Int_t fVolumeID;   //!
    /**  entry position in global frame*/
    TLorentzVector fPosIn;   //!
    /**  exit position in global frame*/
    TLorentzVector fPosOut;   //!
    /**  momentum in*/
    TLorentzVector fMomIn;   //!
    /**  momentum out*/
    TLorentzVector fMomOut;   //!
    /**track time */
    Double_t fTime;   //!
    /**track length */
    Double_t fLength;   //!
    /** energy loss */
    Double_t fELoss;   //!
    /** mass umber*/
    Float_t fA;
    /** atomic number*/
    Float_t fZmat;
    /**  density */
    Float_t fDensity;
    /**radition length */
    Float_t fRadl;
    /**absorption length */
    Float_t fAbsl;

  public:
    /**Add point to collection*/
    void AddPoint(Int_t& ModuleId);
    /**initialize the manager*/
    void Init();
    /**reset*/
    void Reset();
    /**
     * This function is used to access the methods of the class.
     * @return Pointer to the singleton FairRadLenManager object, created
     * with FairRadLenManager::FairRadLenManager().
     */
    static FairRadLenManager* Instance();
};

#endif
