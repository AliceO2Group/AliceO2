/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                    FairRadGridManager source file             -----
// -----          original author                  D.Bertini           -----
// -----          adapted april 2010               O.Hartmann          -----
// -------------------------------------------------------------------------
#ifndef FAIRRADGRIDMANAGER_H
#define FAIRRADGRIDMANAGER_H 1

#include <Rtypes.h>           // for Float_t, Double_t, Int_t, etc
#include <TLorentzVector.h>   // for TLorentzVector
#include <TObjArray.h>        // for TObjArray
#include <TString.h>          // for TString
#include <iostream>           // for basic_ostream::operator<<, etc

class FairMesh;
class TClonesArray;

/**
 * @class FairRadGridManager
 */

class FairRadGridManager
{
  public:
    /**
     * Default constructor.
     * Creates the singleton object of FairRadGridManager class.
     * The pointer to this object can be reached via FairRadGridManager::Instance().
     */
    FairRadGridManager();
    /**
     * Destructor.
     */
    virtual ~FairRadGridManager();

    /**
     * Class definition.
     */
    ClassDef(FairRadGridManager, 1);

  private:
    FairRadGridManager(const FairRadGridManager&);
    FairRadGridManager& operator=(const FairRadGridManager&);

    static FairRadGridManager* fgInstance;

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
    /**estimator*/
    Int_t fEstimator;
    /** the mesh */
    TObjArray* fMeshList;
    /** output file name */
    TString fOutputFileName;

    static Double_t fLtmp;

  public:
    TObjArray* GetMeshList() { return fMeshList; }
    void AddMeshList(TObjArray* list)
    {
        std::cout << " grid manag " << list->GetEntriesFast() << std::endl;
        fMeshList = list;
    }
    Bool_t IsTrackInside(TLorentzVector& vec, FairMesh* aMesh);
    Bool_t IsTrackEntering(TLorentzVector& vec1, TLorentzVector& vec2);
    /** fill the 2D mesh */
    void FillMeshList();
    /**initialize the manager*/
    void Init();
    /**reset*/
    void Reset();
    /**set output file name*/
    void SetOutputFileName(TString tempString) { fOutputFileName = tempString; }
    TString GetOutputFileName() { return fOutputFileName; }
    /**
     * This function is used to access the methods of the class.
     * @return Pointer to the singleton FairRadGridManager object, created
     * with FairRadGridManager::FairRadGridManager().
     */
    static FairRadGridManager* Instance();
};

#endif
