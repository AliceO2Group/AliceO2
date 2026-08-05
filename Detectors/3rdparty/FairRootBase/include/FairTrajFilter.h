/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// ********************************************* //
// ***        D. Kresan   2004-Sep-14        *** //
// ***        D.Kresan@gsi.de                *** //
// ********************************************* //

#ifndef FAIR_TRAJ_FILTER_H
#define FAIR_TRAJ_FILTER_H 1

#include <Rtypes.h>      // for Double_t, Bool_t, Int_t, etc
#include <TGeoTrack.h>   // IWYU pragma: keep needed by cint // TODO: is this still relevant for CINT?
#include <TMCtls.h>      // for multi-threading
#include <TMath.h>       // for Pi, TwoPi
#include <TString.h>     // for TString

class TClonesArray;
class TParticle;

/**
 * @class FairTrajFilter
 * The filter for storing of the trajectories.
 * This singleton class controls storing of trajectories
 * in the gGeoManager list during the simulation.
 * It is created, if FairRun::SetStoreTraj(kTRUE) was called
 * in the run macro
 * before the initialisation. The cuts should be applied
 * after initialisation and before run via
 * FairTrajFilter::Instance()->Set...Cut(...) methods.
 * Three modes of momentum cut (phase space, polar and decart reference systems),
 * are self-excluded. The last that was set, is applied in the simulation.
 * All other cuts are combined together.
 * @author D. Kresan
 * @version 0.1
 * @since 2004-Sep-15
 */

class FairTrajFilter
{
  public:
    /**
     * Default constructor.
     * Creates the singleton object of FairTrajFilter class.
     * The pointer to this object can be reached via FairTrajFilter::Instance().
     */
    FairTrajFilter();

    /**
     * Destructor.
     */
    virtual ~FairTrajFilter();

    /**
     * Class definition.
     */
    ClassDef(FairTrajFilter, 1);

  private:
    FairTrajFilter(const FairTrajFilter&);
    FairTrajFilter& operator=(const FairTrajFilter&);

    static TMCThreadLocal FairTrajFilter* fgInstance;

    Double_t fVxMin;
    Double_t fVxMax;
    Double_t fVyMin;
    Double_t fVyMax;
    Double_t fVzMin;
    Double_t fVzMax;

    Double_t fPMin;
    Double_t fPMax;
    Double_t fThetaMin;
    Double_t fThetaMax;
    Double_t fPhiMin;
    Double_t fPhiMax;

    Double_t fPxMin;
    Double_t fPxMax;
    Double_t fPyMin;
    Double_t fPyMax;
    Double_t fPzMin;
    Double_t fPzMax;

    Double_t fPtMin;
    Double_t fPtMax;
    Double_t fRapidityMin;
    Double_t fRapidityMax;

    Int_t fKinCutType;

    Double_t fEtotMin;
    Double_t fEtotMax;

    Bool_t fStorePrim;
    Bool_t fStoreSec;

    Double_t fStepSizeMin;

    /**
     * collection of tracks
     */
    TClonesArray* fTrackCollection;

    TGeoTrack* fCurrentTrk;

  public:
    TGeoTrack* AddTrack(Int_t trackId, Int_t pdgCode);
    TGeoTrack* AddTrack(TParticle* p);
    TGeoTrack* CheckAddTrack(Int_t trackId, TParticle* p);
    TGeoTrack* GetCurrentTrk() { return fCurrentTrk; }

    void Init(TString brName = "GeoTracks", TString folderName = "MCGeoTrack");
    void Reset();
    /**
     * This function is used to access the methods of the class.
     * @return Pointer to the singleton FairTrajFilter object, created
     * with FairTrajFilter::FairTrajFilter().
     */
    static FairTrajFilter* Instance();

    /**
     * This function applies all availible cuts on different variables.
     * @param p - pointer to the TParticle object.
     * @return kTRUE - if particle survives the cuts. Otherwise - kFALSE.
     */
    Bool_t IsAccepted(const TParticle* p) const;

    /**
     * This function enables the vertex cut.
     * Three modes of momentum cut (phase space, polar and decart reference systems),
     * are self-excluded. The last that was set, is applied in the simulation.
     * @param Box in coordinate space. Only trajectories, created inside this box
     * will be stored. Default values - the cave dimensions.
     */
    void SetVertexCut(Double_t vxMin = -2000.,
                      Double_t vyMin = -2000.,
                      Double_t vzMin = -2000.,
                      Double_t vxMax = 2000.,
                      Double_t vyMax = 2000.,
                      Double_t vzMax = 2000.);

    /**
     * This function enables the momentum cut (polar reference system).
     * Three modes of momentum cut (phase space, polar and decart reference systems),
     * are self-excluded. The last that was set, is applied in the simulation.
     * @param The region in momentum space (polar reference system). Only particles
     * inside this region will be stored.
     * Default values - whole momentum range.
     */
    void SetMomentumCutP(Double_t pMin = 0.,
                         Double_t thetaMin = 0.,
                         Double_t phiMin = 0.,
                         Double_t pMax = 1e10,
                         Double_t thetaMax = TMath::Pi(),
                         Double_t phiMax = TMath::TwoPi());

    /**
     * This function enables the momentum cut (decart reference system).
     * Three modes of momentum cut (phase space, polar and decart reference systems),
     * are self-excluded. The last that was set, is applied in the simulation.
     * @param The region in momentum space (decart reference system). Only particles
     * inside this region will be stored.
     * Default values - whole momentum range.
     */
    void SetMomentumCutD(Double_t pxMin = -1e10,
                         Double_t pyMin = -1e10,
                         Double_t pzMin = -1e10,
                         Double_t pxMax = 1e10,
                         Double_t pyMax = 1e10,
                         Double_t pzMax = 1e10);

    /**
     * This function enables the cut in phase space (pt-rapidity).
     * @param The region in phase space (pt-rapidity). Only particles
     * inside this region will be stored.
     * Default values - whole pt-rapidity range.
     */
    void SetPtRapidityCut(Double_t ptMin = 0.,
                          Double_t ptMax = 1e10,
                          Double_t rapidityMin = -1e10,
                          Double_t rapidityMax = 1e10);

    /**
     * This function enables the cut on total energy.
     * @param Total energy range. Only particles with total energy value
     * inside this range will be stored. Deafult values - whole energy range.
     */
    void SetEnergyCut(Double_t etotMin = 0., Double_t etotMax = 1e10);

    /**
     * This function controls the storing of primaries.
     * @param storePrim - boolean flag to control the storing of primaries.
     */
    inline void SetStorePrimaries(Bool_t storePrim = kTRUE) { fStorePrim = storePrim; };

    /**
     * This function controls the storing of secondaries.
     * @param storeSec - boolean flag to control the storing of secondaries.
     */
    inline void SetStoreSecondaries(Bool_t storeSec = kTRUE) { fStoreSec = storeSec; };

    /**
     * This function controls the process of adding the points to the trajectory.
     * @param stepSizeMin - minimum value of step size, wich will be added to
     * the trajectory.
     */
    void SetStepSizeCut(Double_t stepSizeMin = 0.);

    /**
     * This is the getter for the vertex cut.
     * @param References to the variables where return values should be stored.
     */
    void GetVertexCut(Double_t& vxMin,
                      Double_t& vyMin,
                      Double_t& vzMin,
                      Double_t& vxMax,
                      Double_t& vyMax,
                      Double_t& vzMax) const;

    /**
     * This is the getter for the momentum cut (polar reference system).
     * @param References to the variables where return values should be stored.
     */
    void GetMomentumCutP(Double_t& pMin,
                         Double_t& thetaMin,
                         Double_t& phiMin,
                         Double_t& pMax,
                         Double_t& thetaMax,
                         Double_t& phiMax) const;

    /**
     * This is the getter for the momentum cut (decart reference system).
     * @param References to the variables where return values should be stored.
     */
    void GetMomentumCutD(Double_t& pxMin,
                         Double_t& pyMin,
                         Double_t& pzMin,
                         Double_t& pxMax,
                         Double_t& pyMax,
                         Double_t& pzMax) const;

    /**
     * This is the getter for the space phase cut.
     * @param References to the variables where return values should be stored.
     */
    void GetPtRapidityCut(Double_t& ptMin, Double_t& ptMax, Double_t& rapidityMin, Double_t& rapidityMax) const;

    /**
     * This is the getter for the total energy cut.
     * @param References to the variables where return values should be stored.
     */
    void GetEnergyCut(Double_t& etotMin, Double_t& etotMax) const;

    /**
     * This is the getter for flag of storing of primaries.
     * @return kTRUE if primaries should be stored.
     */
    inline Bool_t IsPrimariesStored() const { return fStorePrim; };

    /**
     * This is the getter for flag of storing of secondaries.
     * @return kTRUE if secondaries should be stored.
     */
    inline Bool_t IsSecondariesStored() const { return fStoreSec; };

    /**
     * This is the getter for the step size cut.
     * @return The minimum value of the step size wich still should be
     * added to the trajectory.
     */
    inline Double_t GetStepSizeCut() const { return fStepSizeMin; };

    TGeoTrack* GetTrack(Int_t trackId);
};

#endif
