// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// An absolut minimal implementation of a TVirtualMC.

#ifndef ALICEO2_MC_TRIVIALMCENGINE_H_
#define ALICEO2_MC_TRIVIALMCENGINE_H_

#include <TVector3.h>
#include <TLorentzVector.h>

#include <TMCProcess.h>

#include <TVirtualMC.h>
#include <TParticle.h>
#include <TVirtualMCStack.h>
#include <TMCManager.h>

namespace o2
{

namespace mc
{

class O2TrivialMCEngine : public TVirtualMC
{

 public:
  O2TrivialMCEngine()
    : TVirtualMC("O2TrivialMCEngine", "O2TrivialMCEngine")
  {
    fApplication->ConstructGeometry();
  }

  /// For now just default destructor
  ~O2TrivialMCEngine() override = default;

  //
  // All the derived stuff
  //

  Bool_t IsRootGeometrySupported() const override
  {
    return kTRUE;
  }

  //
  // functions from GCONS
  // ------------------------------------------------
  //

  void Material(Int_t& kmat, const char* name, Double_t a,
                Double_t z, Double_t dens, Double_t radl, Double_t absl,
                Float_t* buf, Int_t nwbuf) override
  {
    Warning("Material", "Not implemented in this trivial engine");
  }

  void Material(Int_t& kmat, const char* name, Double_t a,
                Double_t z, Double_t dens, Double_t radl, Double_t absl,
                Double_t* buf, Int_t nwbuf) override
  {
    Warning("Material", "Not implemented in this trivial engine");
  }

  void Mixture(Int_t& kmat, const char* name, Float_t* a,
               Float_t* z, Double_t dens, Int_t nlmat, Float_t* wmat) override
  {
    Warning("Mixture", "Not implemented in this trivial engine");
  }

  void Mixture(Int_t& kmat, const char* name, Double_t* a,
               Double_t* z, Double_t dens, Int_t nlmat, Double_t* wmat) override
  {
    Warning("Mixture", "Not implemented in this trivial engine");
  }

  void Medium(Int_t& kmed, const char* name, Int_t nmat,
              Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
              Double_t stemax, Double_t deemax, Double_t epsil,
              Double_t stmin, Float_t* ubuf, Int_t nbuf) override
  {
    Warning("Medium", "Not implemented in this trivial engine");
  }

  void Medium(Int_t& kmed, const char* name, Int_t nmat,
              Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
              Double_t stemax, Double_t deemax, Double_t epsil,
              Double_t stmin, Double_t* ubuf, Int_t nbuf) override
  {
    Warning("Medium", "Not implemented in this trivial engine");
  }

  void Matrix(Int_t& krot, Double_t thetaX, Double_t phiX,
              Double_t thetaY, Double_t phiY, Double_t thetaZ,
              Double_t phiZ) override
  {
    Warning("Matrix", "Not implemented in this trivial engine");
  }

  void Gstpar(Int_t itmed, const char* param, Double_t parval) override
  {
    Warning("Gstpar", "Not implemented in this trivial engine");
  }

  //
  // functions from GGEOM
  // ------------------------------------------------
  //

  Int_t Gsvolu(const char* name, const char* shape, Int_t nmed,
               Float_t* upar, Int_t np) override
  {
    Warning("Gsvolu", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t Gsvolu(const char* name, const char* shape, Int_t nmed,
               Double_t* upar, Int_t np) override
  {
    Warning("Gsvolu", "Not implemented in this trivial engine");
    return -1;
  }

  void Gsdvn(const char* name, const char* mother, Int_t ndiv,
             Int_t iaxis) override
  {
    Warning("Gsdvn", "Not implemented in this trivial engine");
  }

  void Gsdvn2(const char* name, const char* mother, Int_t ndiv,
              Int_t iaxis, Double_t c0i, Int_t numed) override
  {
    Warning("Gsdvn2", "Not implemented in this trivial engine");
  }

  void Gsdvt(const char* name, const char* mother, Double_t step,
             Int_t iaxis, Int_t numed, Int_t ndvmx) override
  {
    Warning("Gsdvt", "Not implemented in this trivial engine");
  }

  void Gsdvt2(const char* name, const char* mother, Double_t step,
              Int_t iaxis, Double_t c0, Int_t numed, Int_t ndvmx) override
  {
    Warning("Gsdvt2", "Not implemented in this trivial engine");
  }

  void Gsord(const char* name, Int_t iax) override
  {
    Warning("Gsord", "Not implemented in this trivial engine");
  }

  void Gspos(const char* name, Int_t nr, const char* mother,
             Double_t x, Double_t y, Double_t z, Int_t irot,
             const char* konly = "ONLY") override
  {
    Warning("Gspos", "Not implemented in this trivial engine");
  }

  void Gsposp(const char* name, Int_t nr, const char* mother,
              Double_t x, Double_t y, Double_t z, Int_t irot,
              const char* konly, Float_t* upar, Int_t np) override
  {
    Warning("Gsposp", "Not implemented in this trivial engine");
  }

  void Gsposp(const char* name, Int_t nr, const char* mother,
              Double_t x, Double_t y, Double_t z, Int_t irot,
              const char* konly, Double_t* upar, Int_t np) override
  {
    Warning("Gsposp", "Not implemented in this trivial engine");
  }

  void Gsbool(const char* onlyVolName, const char* manyVolName) override
  {
    Warning("Gsbool", "Not implemented in this trivial engine");
  }

  //
  // functions for definition of surfaces
  // and material properties for optical physics
  // ------------------------------------------------
  //

  void SetCerenkov(Int_t itmed, Int_t npckov, Float_t* ppckov, Float_t* absco, Float_t* effic, Float_t* rindex, Bool_t aspline = false, Bool_t rspline = false) override
  {
    Warning("SetCerenkov", "Not implemented in this trivial engine");
  }

  void SetCerenkov(Int_t itmed, Int_t npckov, Double_t* ppckov, Double_t* absco, Double_t* effic, Double_t* rindex, Bool_t aspline = false, Bool_t rspline = false) override
  {
    Warning("SetCerenkov", "Not implemented in this trivial engine");
  }

  void DefineOpSurface(const char* name,
                       EMCOpSurfaceModel model,
                       EMCOpSurfaceType surfaceType,
                       EMCOpSurfaceFinish surfaceFinish,
                       Double_t sigmaAlpha) override
  {
    Warning("DefineOpSurface", "Not implemented in this trivial engine");
  }

  void SetBorderSurface(const char* name,
                        const char* vol1Name, int vol1CopyNo,
                        const char* vol2Name, int vol2CopyNo,
                        const char* opSurfaceName) override
  {
    Warning("SetBorderSurface", "Not implemented in this trivial engine");
  }

  void SetSkinSurface(const char* name,
                      const char* volName,
                      const char* opSurfaceName) override
  {
    Warning("SetSkinSurface", "Not implemented in this trivial engine");
  }

  void SetMaterialProperty(Int_t itmed, const char* propertyName, Int_t np, Double_t* pp, Double_t* values, Bool_t createNewKey = false, Bool_t spline = false) override
  {
    Warning("SetMaterialProperty", "Not implemented in this trivial engine");
  }
  void SetMaterialProperty(Int_t itmed, const char* propertyName, Double_t value) override
  {
    Warning("SetMaterialProperty", "Not implemented in this trivial engine");
  }
  void SetMaterialProperty(const char* surfaceName, const char* propertyName, Int_t np, Double_t* pp, Double_t* values, Bool_t createNewKey = false, Bool_t spline = false) override
  {
    Warning("SetMaterialProperty", "Not implemented in this trivial engine");
  }

  //
  // functions for access to geometry
  // ------------------------------------------------
  //

  Bool_t GetTransformation(const TString& volumePath,
                           TGeoHMatrix& matrix) override
  {
    Warning("GetTransformation", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t GetShape(const TString& volumePath,
                  TString& shapeType, TArrayD& par) override
  {
    Warning("GetShape", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t GetMaterial(Int_t imat, TString& name,
                     Double_t& a, Double_t& z, Double_t& density,
                     Double_t& radl, Double_t& inter, TArrayD& par) override
  {
    Warning("GetMaterial", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t GetMaterial(const TString& volumeName,
                     TString& name, Int_t& imat,
                     Double_t& a, Double_t& z, Double_t& density,
                     Double_t& radl, Double_t& inter, TArrayD& par) override
  {
    Warning("GetMaterial", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t GetMedium(const TString& volumeName,
                   TString& name, Int_t& imed,
                   Int_t& nmat, Int_t& isvol, Int_t& ifield,
                   Double_t& fieldm, Double_t& tmaxfd, Double_t& stemax,
                   Double_t& deemax, Double_t& epsil, Double_t& stmin,
                   TArrayD& par) override
  {
    Warning("GetMedium", "Not implemented in this trivial engine");
    return kFALSE;
  }

  void WriteEuclid(const char* filnam, const char* topvol,
                   Int_t number, Int_t nlevel) override
  {
    Warning("WriteEuclid", "Not implemented in this trivial engine");
  }

  void SetRootGeometry() override {}

  void SetUserParameters(Bool_t isUserParameters) override
  {
    Warning("SetUserParameters", "Not implemented in this trivial engine");
  }

  //
  // get methods
  // ------------------------------------------------
  //

  Int_t VolId(const char* volName) const override
  {
    Warning("VolId", "Not implemented in this trivial engine");
    return -1;
  }

  const char* VolName(Int_t id) const override
  {
    Warning("VolName", "Not implemented in this trivial engine");
    return "";
  }

  Int_t MediumId(const char* mediumName) const override
  {
    Warning("MediumId", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t NofVolumes() const override
  {
    Warning("NofVolumes", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t VolId2Mate(Int_t id) const override
  {
    Warning("VolId2Mate", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t NofVolDaughters(const char* volName) const override
  {
    Warning("NofVolDaughters", "Not implemented in this trivial engine");
    return -1;
  }

  const char* VolDaughterName(const char* volName, Int_t i) const override
  {
    Warning("VolDaughterName", "Not implemented in this trivial engine");
    return "";
  }

  Int_t VolDaughterCopyNo(const char* volName, Int_t i) const override
  {
    Warning("VolDaughterCopyNo", "Not implemented in this trivial engine");
    return -1;
  }

  //
  // ------------------------------------------------
  // methods for sensitive detectors
  // ------------------------------------------------
  //

  // Set a sensitive detector to a volume
  // - volName - the volume name
  // - sd - the user sensitive detector
  void SetSensitiveDetector(const TString& volName, TVirtualMCSensitiveDetector* sd) override
  {
    Warning("SetSensitiveDetector", "Not implemented in this trivial engine");
  }

  // Get a sensitive detector of a volume
  // - volName - the volume name
  TVirtualMCSensitiveDetector* GetSensitiveDetector(const TString& volName) const override
  {
    Warning("GetSensitiveDetector", "Not implemented in this trivial engine");
    return nullptr;
  }

  // The scoring option:
  // if true, scoring is performed only via user defined sensitive detectors and
  // MCApplication::Stepping is not called
  void SetExclusiveSDScoring(Bool_t exclusiveSDScoring) override
  {
    Warning("SetExclusiveSDScoring", "Not implemented in this trivial engine");
  }

  //
  // ------------------------------------------------
  // methods for physics management
  // ------------------------------------------------
  //

  //
  // set methods
  // ------------------------------------------------
  //

  Bool_t SetCut(const char* cutName, Double_t cutValue) override
  {
    Warning("SetCut", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t SetProcess(const char* flagName, Int_t flagValue) override
  {
    Warning("SetProcess", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t DefineParticle(Int_t pdg, const char* name,
                        TMCParticleType mcType,
                        Double_t mass, Double_t charge, Double_t lifetime) override
  {
    Warning("DefineParticle", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t DefineParticle(Int_t pdg, const char* name,
                        TMCParticleType mcType,
                        Double_t mass, Double_t charge, Double_t lifetime,
                        const TString& pType, Double_t width,
                        Int_t iSpin, Int_t iParity, Int_t iConjugation,
                        Int_t iIsospin, Int_t iIsospinZ, Int_t gParity,
                        Int_t lepton, Int_t baryon,
                        Bool_t stable, Bool_t shortlived = kFALSE,
                        const TString& subType = "",
                        Int_t antiEncoding = 0, Double_t magMoment = 0.0,
                        Double_t excitation = 0.0) override
  {
    Warning("DefineParticle", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t DefineIon(const char* name, Int_t Z, Int_t A,
                   Int_t Q, Double_t excEnergy, Double_t mass = 0.) override
  {
    Warning("DefineIon", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t SetDecayMode(Int_t pdg, Float_t bratio[6], Int_t mode[6][3]) override
  {
    Warning("SetDecayMode", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Double_t Xsec(char*, Double_t, Int_t, Int_t) override
  {
    Warning("Xsec", "Not implemented in this trivial engine");
    return -1.;
  }

  //
  // particle table usage
  // ------------------------------------------------
  //

  Int_t IdFromPDG(Int_t pdg) const override
  {
    Warning("IdFromPDG", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t PDGFromId(Int_t id) const override
  {
    Warning("PDGFromId", "Not implemented in this trivial engine");
    return -1;
  }

  //
  // get methods
  // ------------------------------------------------
  //

  TString ParticleName(Int_t pdg) const override
  {
    Warning("ParticleName", "Not implemented in this trivial engine");
    return TString();
  }

  Double_t ParticleMass(Int_t pdg) const override
  {
    Warning("ParticleMass", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t ParticleCharge(Int_t pdg) const override
  {
    Warning("ParticleCharge", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t ParticleLifeTime(Int_t pdg) const override
  {
    Warning("ParticleLifeTime", "Not implemented in this trivial engine");
    return -1.;
  }

  TMCParticleType ParticleMCType(Int_t pdg) const override
  {
    Warning("ParticleMCType", "Not implemented in this trivial engine");
    return TMCParticleType();
  }
  //
  // ------------------------------------------------
  // methods for step management
  // ------------------------------------------------
  //

  //
  // action methods
  // ------------------------------------------------
  //

  void StopTrack() override
  {
    Warning("StopTrack", "Not implemented in this trivial engine");
  }

  void StopEvent() override
  {
    Warning("StopEvent", "Not implemented in this trivial engine");
  }

  void StopRun() override
  {
    Warning("StopRun", "Not implemented in this trivial engine");
  }

  //
  // set methods
  // ------------------------------------------------
  //

  void SetMaxStep(Double_t) override
  {
    Warning("SetMaxStep", "Not implemented in this trivial engine");
  }

  void SetMaxNStep(Int_t) override
  {
    Warning("SetMaxNStep", "Not implemented in this trivial engine");
  }

  void SetUserDecay(Int_t pdg) override
  {
    Warning("SetUserDecay", "Not implemented in this trivial engine");
  }

  void ForceDecayTime(Float_t) override
  {
    Warning("ForceDecayTime", "Not implemented in this trivial engine");
  }

  //
  // tracking volume(s)
  // ------------------------------------------------
  //

  Int_t CurrentVolID(Int_t& copyNo) const override
  {
    Warning("CurrentVolID", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t CurrentVolOffID(Int_t off, Int_t& copyNo) const override
  {
    Warning("CurrentVolOffID", "Not implemented in this trivial engine");
    return -1;
  }

  const char* CurrentVolName() const override
  {
    Warning("CurrentVolName", "Not implemented in this trivial engine");
    return "";
  }

  const char* CurrentVolOffName(Int_t off) const override
  {
    Warning("CurrentVolOffName", "Not implemented in this trivial engine");
    return "";
  }

  const char* CurrentVolPath() override
  {
    Warning("CurrentVolPath", "Not implemented in this trivial engine");
    return "";
  }

  Bool_t CurrentBoundaryNormal(
    Double_t& x, Double_t& y, Double_t& z) const override
  {
    Warning("CurrentBoundaryNormal", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Int_t CurrentMaterial(Float_t& a, Float_t& z,
                        Float_t& dens, Float_t& radl, Float_t& absl) const override
  {
    Warning("CurrentMaterial", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t CurrentMedium() const override
  {
    Warning("CurrentMedium", "Not implemented in this trivial engine");
    return -1;
  }

  Int_t CurrentEvent() const override
  {
    Warning("CurrentEvent", "Not implemented in this trivial engine");
    return -1;
  }

  void Gmtod(Float_t* xm, Float_t* xd, Int_t iflag) override
  {
    Warning("Gmtod", "Not implemented in this trivial engine");
  }

  void Gmtod(Double_t* xm, Double_t* xd, Int_t iflag) override
  {
    Warning("Gmtod", "Not implemented in this trivial engine");
  }

  void Gdtom(Float_t* xd, Float_t* xm, Int_t iflag) override
  {
    Warning("Gdtom", "Not implemented in this trivial engine");
  }

  void Gdtom(Double_t* xd, Double_t* xm, Int_t iflag) override
  {
    Warning("Gdtom", "Not implemented in this trivial engine");
  }

  Double_t MaxStep() const override
  {
    Warning("MaxStep", "Not implemented in this trivial engine");
    return -1.;
  }

  Int_t GetMaxNStep() const override
  {
    Warning("GetMaxNStep", "Not implemented in this trivial engine");
    return -1;
  }

  //
  // get methods
  // tracking particle
  // dynamic properties
  // ------------------------------------------------
  //

  void TrackPosition(TLorentzVector& position) const override
  {
    Warning("TrackPosition", "Not implemented in this trivial engine");
  }

  void TrackPosition(Double_t& x, Double_t& y, Double_t& z) const override
  {
    Warning("TrackPosition", "Not implemented in this trivial engine");
  }

  void TrackPosition(Float_t& x, Float_t& y, Float_t& z) const override
  {
    Warning("TrackPosition", "Not implemented in this trivial engine");
  }

  void TrackMomentum(TLorentzVector& momentum) const override
  {
    Warning("TrackMomentum", "Not implemented in this trivial engine");
  }

  void TrackMomentum(Double_t& px, Double_t& py, Double_t& pz, Double_t& etot) const override
  {
    Warning("TrackMomentum", "Not implemented in this trivial engine");
  }

  void TrackMomentum(Float_t& px, Float_t& py, Float_t& pz, Float_t& etot) const override
  {
    Warning("TrackMomentum", "Not implemented in this trivial engine");
  }

  Double_t TrackStep() const override
  {
    Warning("TrackStep", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t TrackLength() const override
  {
    Warning("TrackLength", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t TrackTime() const override
  {
    Warning("TrackTime", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t Edep() const override
  {
    Warning("Edep", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t NIELEdep() const override
  {
    Warning("NIELEdep", "Not implemented in this trivial engine");
    return -1.;
  }

  Int_t StepNumber() const override
  {
    Warning("StepNumber", "Not implemented in this trivial engine");
    return -1;
  }

  Double_t TrackWeight() const override
  {
    Warning("TrackWeight", "Not implemented in this trivial engine");
    return -1.;
  }

  void TrackPolarization(Double_t& polX, Double_t& polY, Double_t& polZ) const override
  {
    Warning("TrackPolarization", "Not implemented in this trivial engine");
  }

  void TrackPolarization(TVector3& pol) const override
  {
    Warning("TrackPolarization", "Not implemented in this trivial engine");
  }

  //
  // get methods
  // tracking particle
  // static properties
  // ------------------------------------------------
  //

  Int_t TrackPid() const override
  {
    Warning("TrackPid", "Not implemented in this trivial engine");
    return -1;
  }

  Double_t TrackCharge() const override
  {
    Warning("TrackCharge", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t TrackMass() const override
  {
    Warning("TrackMass", "Not implemented in this trivial engine");
    return -1.;
  }

  Double_t Etot() const override
  {
    Warning("Etot", "Not implemented in this trivial engine");
    return -1.;
  }

  //
  // get methods - track status
  // ------------------------------------------------
  //

  Bool_t IsNewTrack() const override
  {
    Warning("IsNewTrack", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackInside() const override
  {
    Warning("IsTrackInside", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackEntering() const override
  {
    Warning("IsTrackEntering", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackExiting() const override
  {
    Warning("IsTrackExiting", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackOut() const override
  {
    Warning("IsTrackOut", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackDisappeared() const override
  {
    Warning("IsTrackDisappeared", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackStop() const override
  {
    Warning("IsTrackStop", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsTrackAlive() const override
  {
    Warning("IsTrackAlive", "Not implemented in this trivial engine");
    return kFALSE;
  }

  //
  // get methods - secondaries
  // ------------------------------------------------
  //

  Int_t NSecondaries() const override
  {
    Warning("NSecondaries", "Not implemented in this trivial engine");
    return -1;
  }

  void GetSecondary(Int_t isec, Int_t& particleId,
                    TLorentzVector& position, TLorentzVector& momentum) override
  {
    Warning("GetSecondary", "Not implemented in this trivial engine");
  }

  TMCProcess ProdProcess(Int_t isec) const override
  {
    Warning("ProdProcess", "Not implemented in this trivial engine");
    return TMCProcess();
  }

  Int_t StepProcesses(TArrayI& proc) const override
  {
    Warning("StepProcesses", "Not implemented in this trivial engine");
    return -1;
  }

  Bool_t SecondariesAreOrdered() const override
  {
    Warning("SecondariesAreOrdered", "Not implemented in this trivial engine");
    return kFALSE;
  }

  //
  // ------------------------------------------------
  // Control methods
  // ------------------------------------------------
  //

  void Init() override
  {
    fApplication->InitGeometry();
    Warning("Init", "Not implemented in this trivial engine");
  }

  void BuildPhysics() override
  {
    Warning("BuildPhysics", "Not implemented in this trivial engine");
  }

  void ProcessEvent(Int_t eventId) override
  {
    processEventImpl();
  }

  void ProcessEvent(Int_t eventId, Bool_t isInterruptible) override
  {
    Warning("ProcessEvent", "Not implemented in this trivial engine");
  }

  void ProcessEvent() override
  {
    processEventImpl();
  }

  void InterruptTrack() override
  {
    Info("InterruptTrack", "Not implemented in this trivial engine");
  }

  Bool_t ProcessRun(Int_t nevent) override
  {
    if (nevent <= 0) {
      return kFALSE;
    }

    for (Int_t i = 0; i < nevent; i++) {
      ProcessEvent(i);
    }
    return kTRUE;
  }

  void TerminateRun() override
  {
    Warning("TerminateRun", "Not implemented in this trivial engine");
  }

  void InitLego() override
  {
    Warning("InitLego", "Not implemented in this trivial engine");
  }

  void SetCollectTracks(Bool_t collectTracks) override
  {
    Warning("SetCollectTracks", "Not implemented in this trivial engine");
  }

  Bool_t IsCollectTracks() const override
  {
    Warning("IsCollectTracks", "Not implemented in this trivial engine");
    return kFALSE;
  }

  Bool_t IsMT() const override { return kFALSE; }

 private:
  O2TrivialMCEngine(O2TrivialMCEngine const&);
  O2TrivialMCEngine& operator=(O2TrivialMCEngine const&);
  void processEventImpl()
  {
    auto stack = GetStack();
    if (!TMCManager::Instance()) {
      fApplication->GeneratePrimaries();
    }
    Int_t nPopped{};
    Int_t itrack;
    fApplication->BeginEvent();
    while (true) {
      if (!stack->PopNextTrack(itrack)) {
        break;
      }
      nPopped++;
      fApplication->BeginPrimary();
      fApplication->PreTrack();
      fApplication->PostTrack();
      fApplication->FinishPrimary();
    }
    fApplication->FinishEvent();
    Info("processEventImpl", "Popped %d primaries", nPopped);
  }
};

} // namespace mc

} // namespace o2

#endif
