/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairRadMapManager source file            -----
// -------------------------------------------------------------------------
#ifndef FAIRRADMAPMANAGER_H
#define FAIRRADMAPMANAGER_H 1

#include <Rtypes.h>           // for Double_t, Float_t, Int_t, etc
#include <TLorentzVector.h>   // for TLorentzVector

class TClonesArray;
class TMap;

/**
 * @class FairRadMapManager
 */

class FairRadMapManager
{
  public:
    /**
     * Default constructor.
     * Creates the singleton object of FairRadMapManager class.
     * The pointer to this object can be reached via FairRadMapManager::Instance().
     */
    FairRadMapManager();

    /**
     * Destructor.
     */
    virtual ~FairRadMapManager();

    /**
     * Class definition.
     */
    ClassDef(FairRadMapManager, 1);

  private:
    FairRadMapManager(const FairRadMapManager&);
    FairRadMapManager& operator=(const FairRadMapManager&);

    static FairRadMapManager* fgInstance;

    /**
     * collection of point
     */
    TClonesArray* fPointCollection;
    /**track index */
    Int_t fTrackID;   //!
    /**volume id */
    Int_t fVolumeID;   //!
    /** pid */
    Int_t fPdg;   //!
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
    /**track setp */
    Double_t fStep;
    /** energy loss */
    Double_t fELoss;   //!
    /**dose */
    Double_t fDose;
    Double_t fDoseSL;
    /** mass umber*/
    Float_t fA;
    /** atomic number*/
    Float_t fZmat;
    /** radiation length*/
    Float_t fRadl;
    /**  density */
    Float_t fDensity;
    /**absorption length */
    Float_t fAbsl;
    //**volume, mass */
    Double_t fActVol;
    Double_t fActMass;

    TMap* fMassMap;

  public:
    /**Add point to collection*/
    void AddPoint(Int_t& ModuleId);
    /**initialize the manager*/
    void Init();
    /**reset*/
    void Reset();
    /**
     * This function is used to access the methods of the class.
     * @return Pointer to the singleton FairRadMapManager object, created
     * with FairRadMapManager::FairRadMapManager().
     */
    static FairRadMapManager* Instance();
    //  void GetGeoManager();
};

#endif
