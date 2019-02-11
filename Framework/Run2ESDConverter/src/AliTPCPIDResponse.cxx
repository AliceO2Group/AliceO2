    /**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

//-----------------------------------------------------------------
//           Implementation of the TPC PID class
// Very naive one... Should be made better by the detector experts...
//      Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
// With many additions and modifications suggested by
//      Alexander Kalweit, GSI, alexander.philipp.kalweit@cern.ch
//      Dariusz Miskowiec, GSI, D.Miskowiec@gsi.de
//      Jens Wiechula, IKF, Jens.Wiechula@cern.ch
// ...and some modifications by
//      Mikolaj Krzewicki, GSI, mikolaj.krzewicki@cern.ch
// ...and some modifications plus eta correction functions by
//      Benjamin Hess, University of Tuebingen, bhess@cern.ch
//-----------------------------------------------------------------

#include <TGraph.h>
#include <TError.h>
#include <TObjArray.h>
#include <TSpline.h>
#include <TBits.h>
#include <TMath.h>
#include <TH2D.h>
#include <TSystem.h>
#include <TMD5.h>

#include <AliLog.h>
#include "AliExternalTrackParam.h"
#include "AliVTrack.h"
#include "AliTPCPIDResponse.h"
#include "AliTPCdEdxInfo.h"
#include "AliOADBContainer.h"
#include "TFile.h"
#include "TSpline.h"

ClassImp(AliTPCPIDResponse);


AliTPCPIDResponse *AliTPCPIDResponse::fgInstance =0;

const char* AliTPCPIDResponse::fgkGainScenarioName[fgkNumberOfGainScenarios+1]=
{
  "", //default - no name
  "1", //all high
  "2", //OROC only
  "unknown"
};

//_________________________________________________________________________
AliTPCPIDResponse::AliTPCPIDResponse():
  TNamed(),
  fMIP(50.),
  fRes0(),
  fResN2(),
  fKp1(0.0283086),
  fKp2(2.63394e+01),
  fKp3(5.04114e-11),
  fKp4(2.12543),
  fKp5(4.88663),
  fUseDatabase(kFALSE),
  fResponseFunctions(fgkNumberOfParticleSpecies*fgkNumberOfGainScenarios),
  fOADBContainer(0x0),
  fVoltageMap(72),
  fLowGainIROCthreshold(-40),
  fBadIROCthreshhold(-70),
  fLowGainOROCthreshold(-40),
  fBadOROCthreshhold(-40),
  fMaxBadLengthFraction(0.5),
  fMagField(0.),
  fhEtaCorr(0x0),
  fhEtaSigmaPar1(0x0),
  fSigmaPar0(0.0),
  fCurrentEventMultiplicity(0),
  fCorrFuncSlope(0x0),
  fCorrFuncCurv(0x0),
  fIsNewPbPbParam(kFALSE),
  fCorrFuncMultiplicity(0x0),
  fCorrFuncMultiplicityTanTheta(0x0),
  fCorrFuncSigmaMultiplicity(0x0),
  fdEdxType(kdEdxTrack),
  fdEdxChargeType(0),
  fdEdxWeightType(0),
  fIROCweight(1.),
  fOROCmedWeight(1.),
  fOROClongWeight(1.),
  fRecoPassNameUsed(),
  fSplineArray()
{
  //
  //  The default constructor
  //
  
  AliLog::SetClassDebugLevel("AliTPCPIDResponse", AliLog::kInfo); 
  
  for (Int_t i=0; i<fgkNumberOfGainScenarios; i++) {fRes0[i]=0.07;fResN2[i]=0.0;}
  
  fCorrFuncSlope = new TF1("fCorrFuncSlope", "[0] + [1] * x",0,0.2);
  fCorrFuncCurv = new TF1("fCorrFuncCurv","[0] + [1] * x",0,0.2);
  
  fCorrFuncMultiplicity = new TF1("fCorrFuncMultiplicity", 
                                  "[0] + [1]*TMath::Max([4], TMath::Min(x, [3])) + [2] * TMath::Power(TMath::Max([4], TMath::Min(x, [3])), 2)",
                                  0., 0.2);
  fCorrFuncMultiplicityTanTheta = new TF1("fCorrFuncMultiplicityTanTheta", "[0] * (x - [2]) + [1] * (x * x - [2] * [2])", -1.5, 1.5);
  fCorrFuncSigmaMultiplicity = new TF1("fCorrFuncSigmaMultiplicity",
                                       "TMath::Max(0.0, [0] + [1]*TMath::Min(x, [3]) + [2] * TMath::Power(TMath::Min(x, [3]), 2))", 0., 0.2);

  
  ResetMultiplicityCorrectionFunctions();
  fgInstance=this;
}
/*TODO remove?
//_________________________________________________________________________
AliTPCPIDResponse::AliTPCPIDResponse(const Double_t *param):
  TNamed(),
  fMIP(param[0]),
  fRes0(),
  fResN2(),
  fKp1(0.0283086),
  fKp2(2.63394e+01),
  fKp3(5.04114e-11),
  fKp4(2.12543),
  fKp5(4.88663),
  fUseDatabase(kFALSE),
  fResponseFunctions(fgkNumberOfParticleSpecies*fgkNumberOfGainScenarios),
  fVoltageMap(72),
  fLowGainIROCthreshold(-40),
  fBadIROCthreshhold(-70),
  fLowGainOROCthreshold(-40),
  fBadOROCthreshhold(-40),
  fMaxBadLengthFraction(0.5),
  fMagField(0.),
  fhEtaCorr(0x0),
  fhEtaSigmaPar1(0x0),
  fSigmaPar0(0.0)
{
  //
  //  The main constructor
  //
  for (Int_t i=0; i<fgkNumberOfGainScenarios; i++) {fRes0[i]=param[1];fResN2[i]=param[2];}
}
*/

//_________________________________________________________________________
AliTPCPIDResponse::~AliTPCPIDResponse()
{
  //
  // Destructor
  //
  
  delete fhEtaCorr;
  fhEtaCorr = 0x0;
  
  delete fhEtaSigmaPar1;
  fhEtaSigmaPar1 = 0x0;
  
  delete fCorrFuncMultiplicity;
  fCorrFuncMultiplicity = 0x0;
  
  delete fCorrFuncMultiplicityTanTheta;
  fCorrFuncMultiplicityTanTheta = 0x0;
  
  delete fCorrFuncSigmaMultiplicity;
  fCorrFuncSigmaMultiplicity = 0x0;
  if (fgInstance==this) fgInstance=0;

  delete fOADBContainer;
}


//_________________________________________________________________________
AliTPCPIDResponse::AliTPCPIDResponse(const AliTPCPIDResponse& that):
  TNamed(that),
  fMIP(that.fMIP),
  fRes0(),
  fResN2(),
  fKp1(that.fKp1),
  fKp2(that.fKp2),
  fKp3(that.fKp3),
  fKp4(that.fKp4),
  fKp5(that.fKp5),
  fUseDatabase(that.fUseDatabase),
  fResponseFunctions(that.fResponseFunctions),
  fOADBContainer(0x0),
  fVoltageMap(that.fVoltageMap),
  fLowGainIROCthreshold(that.fLowGainIROCthreshold),
  fBadIROCthreshhold(that.fBadIROCthreshhold),
  fLowGainOROCthreshold(that.fLowGainOROCthreshold),
  fBadOROCthreshhold(that.fBadOROCthreshhold),
  fMaxBadLengthFraction(that.fMaxBadLengthFraction),
  fMagField(that.fMagField),
  fhEtaCorr(0x0),
  fhEtaSigmaPar1(0x0),
  fSigmaPar0(that.fSigmaPar0),
  fCorrFuncSlope(0x0),
  fCorrFuncCurv(0x0),
  fIsNewPbPbParam(that.fIsNewPbPbParam),
  fCurrentEventMultiplicity(that.fCurrentEventMultiplicity),
  fCorrFuncMultiplicity(0x0),
  fCorrFuncMultiplicityTanTheta(0x0),
  fCorrFuncSigmaMultiplicity(0x0),
  fdEdxType(kdEdxTrack),
  fdEdxChargeType(that.fdEdxChargeType),
  fdEdxWeightType(that.fdEdxWeightType),
  fIROCweight(that.fIROCweight),
  fOROCmedWeight(that.fOROCmedWeight),
  fOROClongWeight(that.fOROClongWeight),
  fRecoPassNameUsed(that.fRecoPassNameUsed),
  fSplineArray()
{
  //copy ctor
  for (Int_t i=0; i<fgkNumberOfGainScenarios; i++) {fRes0[i]=that.fRes0[i];fResN2[i]=that.fResN2[i];}
 
  // Copy eta maps
  if (that.fhEtaCorr) {
    fhEtaCorr = new TH2D(*(that.fhEtaCorr));
    fhEtaCorr->SetDirectory(0);
  }
  
  if (that.fhEtaSigmaPar1) {
    fhEtaSigmaPar1 = new TH2D(*(that.fhEtaSigmaPar1));
    fhEtaSigmaPar1->SetDirectory(0);
  }
  
  // Copy multiplicity correction functions
  if (that.fCorrFuncSlope) {
    fCorrFuncSlope = new TF1(*(that.fCorrFuncSlope));
  }
  
  if (that.fCorrFuncCurv) {
    fCorrFuncCurv = new TF1(*(that.fCorrFuncCurv));
  }  
  
  if (that.fCorrFuncMultiplicity) {
    fCorrFuncMultiplicity = new TF1(*(that.fCorrFuncMultiplicity));
  }
  
  if (that.fCorrFuncMultiplicityTanTheta) {
    fCorrFuncMultiplicityTanTheta = new TF1(*(that.fCorrFuncMultiplicityTanTheta));
  }
  
  if (that.fCorrFuncSigmaMultiplicity) {
    fCorrFuncSigmaMultiplicity = new TF1(*(that.fCorrFuncSigmaMultiplicity));
  }
}

//_________________________________________________________________________
AliTPCPIDResponse& AliTPCPIDResponse::operator=(const AliTPCPIDResponse& that)
{
  //assignment
  if (&that==this) return *this;
  TNamed::operator=(that);
  fMIP=that.fMIP;
  fKp1=that.fKp1;
  fKp2=that.fKp2;
  fKp3=that.fKp3;
  fKp4=that.fKp4;
  fKp5=that.fKp5;
  fUseDatabase=that.fUseDatabase;
  fResponseFunctions=that.fResponseFunctions;
  fOADBContainer=0x0;
  fVoltageMap=that.fVoltageMap;
  fLowGainIROCthreshold=that.fLowGainIROCthreshold;
  fBadIROCthreshhold=that.fBadIROCthreshhold;
  fLowGainOROCthreshold=that.fLowGainOROCthreshold;
  fBadOROCthreshhold=that.fBadOROCthreshhold;
  fMaxBadLengthFraction=that.fMaxBadLengthFraction;
  fMagField=that.fMagField;
  fCurrentEventMultiplicity=that.fCurrentEventMultiplicity;
  for (Int_t i=0; i<fgkNumberOfGainScenarios; i++) {fRes0[i]=that.fRes0[i];fResN2[i]=that.fResN2[i];}

  delete fhEtaCorr;
  fhEtaCorr=0x0;
  if (that.fhEtaCorr) {
    fhEtaCorr = new TH2D(*(that.fhEtaCorr));
    fhEtaCorr->SetDirectory(0);
  }
  
  delete fhEtaSigmaPar1;
  fhEtaSigmaPar1=0x0;
  if (that.fhEtaSigmaPar1) {
    fhEtaSigmaPar1 = new TH2D(*(that.fhEtaSigmaPar1));
    fhEtaSigmaPar1->SetDirectory(0);
  }
  
  fSigmaPar0 = that.fSigmaPar0;
  
  delete fCorrFuncSlope;
  fCorrFuncSlope = 0x0;
  if (that.fCorrFuncSlope) {
    fCorrFuncSlope = new TF1(*(that.fCorrFuncSlope));
  }  
  
  delete fCorrFuncCurv;
  fCorrFuncCurv = 0x0;
  if (that.fCorrFuncCurv) {
    fCorrFuncCurv = new TF1(*(that.fCorrFuncCurv));
  }    
  
  delete fCorrFuncMultiplicity;
  fCorrFuncMultiplicity = 0x0;
  if (that.fCorrFuncMultiplicity) {
    fCorrFuncMultiplicity = new TF1(*(that.fCorrFuncMultiplicity));
  }
  
  delete fCorrFuncMultiplicityTanTheta;
  fCorrFuncMultiplicityTanTheta = 0x0;
  if (that.fCorrFuncMultiplicityTanTheta) {
    fCorrFuncMultiplicityTanTheta = new TF1(*(that.fCorrFuncMultiplicityTanTheta));
  }
  
  delete fCorrFuncSigmaMultiplicity;
  fCorrFuncSigmaMultiplicity = 0x0;
  if (that.fCorrFuncSigmaMultiplicity) {
    fCorrFuncSigmaMultiplicity = new TF1(*(that.fCorrFuncSigmaMultiplicity));
  }

  fdEdxType      =that.fdEdxType;
  fdEdxChargeType=that.fdEdxChargeType;
  fdEdxWeightType=that.fdEdxWeightType;
  fIROCweight    =that.fIROCweight;
  fOROCmedWeight =that.fOROCmedWeight;
  fOROClongWeight=that.fOROClongWeight;
  fRecoPassNameUsed=that.fRecoPassNameUsed;

  return *this;
}

//_________________________________________________________________________
Double_t AliTPCPIDResponse::Bethe(Double_t betaGamma) const {
  //
  // This is the Bethe-Bloch function normalised to 1 at the minimum
  // WARNING
  // Simulated and reconstructed Bethe-Bloch differs
  //           Simulated  curve is the dNprim/dx
  //           Reconstructed is proportianal dNtot/dx
  // Temporary fix for production -  Simple linear correction function
  // Future    2 Bethe Bloch formulas needed
  //           1. for simulation
  //           2. for reconstructed PID
  //
  
//   const Float_t kmeanCorrection =0.1;
  Double_t bb=
    AliExternalTrackParam::BetheBlochAleph(betaGamma,fKp1,fKp2,fKp3,fKp4,fKp5);
  return bb*fMIP;
}

//_________________________________________________________________________
void AliTPCPIDResponse::SetBetheBlochParameters(Double_t kp1,
                             Double_t kp2,
                             Double_t kp3,
                             Double_t kp4,
                             Double_t kp5) {
  //
  // Set the parameters of the ALEPH Bethe-Bloch formula
  //
  fKp1=kp1;
  fKp2=kp2;
  fKp3=kp3;
  fKp4=kp4;
  fKp5=kp5;
}

//_________________________________________________________________________
void AliTPCPIDResponse::SetSigma(Float_t res0, Float_t resN2) {
  //
  // Set the relative resolution  sigma_rel = res0 * sqrt(1+resN2/npoint)
  //
  for (Int_t i=0; i<fgkNumberOfGainScenarios; i++) {fRes0[i]=res0;fResN2[i]=resN2;}
}

//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSignal(Float_t mom,
					      AliPID::EParticleType n) const {
  //
  // Deprecated function (for backward compatibility). Please use 
  // GetExpectedSignal(const AliVTrack* track, AliPID::EParticleType species, ETPCdEdxSource dedxSource,
  //                   Bool_t correctEta, Bool_t correctMultiplicity);
  // instead!
  //
  //
  // Calculates the expected PID signal as the function of 
  // the information stored in the track, for the specified particle type 
  //  
  // At the moment, these signals are just the results of calling the 
  // Bethe-Bloch formula. 
  // This can be improved. By taking into account the number of
  // assigned clusters and/or the track dip angle, for example.  
  //
  
  //charge factor. BB goes with z^2, however in reality it is slightly larger (calibration, threshold effects, ...)
  // !!! Splines for light nuclei need to be normalised to this factor !!!
  const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(n),2.3);
  
  Double_t mass=AliPID::ParticleMassZ(n);
  if (!fUseDatabase) return Bethe(mom/mass) * chargeFactor;
  //
  const TSpline3 * responseFunction = (TSpline3 *) fResponseFunctions.UncheckedAt(n);

  if (!responseFunction) return Bethe(mom/mass) * chargeFactor;
  
  return fMIP*responseFunction->Eval(mom/mass)*chargeFactor;

}

//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSigma(Float_t mom, 
                                             Int_t nPoints,
                                             AliPID::EParticleType n) const {
  //
  // Deprecated function (for backward compatibility). Please use 
  // GetExpectedSigma(onst AliVTrack* track, AliPID::EParticleType species, 
  // ETPCdEdxSource dedxSource, Bool_t correctEta) instead!
  //
  //
  // Calculates the expected sigma of the PID signal as the function of 
  // the information stored in the track, for the specified particle type 
  //  
  
  if (nPoints != 0) 
    return GetExpectedSignal(mom,n)*fRes0[0]*sqrt(1. + fResN2[0]/nPoints);
  else
    return GetExpectedSignal(mom,n)*fRes0[0];
}

////////////////////////////////////////////////////NEW//////////////////////////////

//_________________________________________________________________________
void AliTPCPIDResponse::SetSigma(Float_t res0, Float_t resN2, ETPCgainScenario gainScenario) {
  //
  // Set the relative resolution  sigma_rel = res0 * sqrt(1+resN2/npoint)
  //
  if ((Int_t)gainScenario>=(Int_t)fgkNumberOfGainScenarios) return; //TODO: better handling!
  fRes0[gainScenario]=res0;
  fResN2[gainScenario]=resN2;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSignal(const AliVTrack* track,
                                              AliPID::EParticleType species,
                                              Double_t /*dEdx*/,
                                              const TSpline3* responseFunction,
                                              Bool_t correctEta,
                                              Bool_t correctMultiplicity) const 
{
  // Calculates the expected PID signal as the function of 
  // the information stored in the track and the given parameters,
  // for the specified particle type 
  //  
  // At the moment, these signals are just the results of calling the 
  // Bethe-Bloch formula plus, if desired, taking into account the eta dependence
  // and the multiplicity dependence (for PbPb). 
  // This can be improved. By taking into account the number of
  // assigned clusters and/or the track dip angle, for example.  
  //
  
  Double_t mom=track->GetTPCmomentum();
  Double_t mass=AliPID::ParticleMassZ(species);
  
  //charge factor. BB goes with z^2, however in reality it is slightly larger (calibration, threshold effects, ...)
  // !!! Splines for light nuclei need to be normalised to this factor !!!
  const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(species),2.3);
  
  if (!responseFunction)
    return Bethe(mom/mass) * chargeFactor;
  
  Double_t dEdxSplines = fMIP*responseFunction->Eval(mom/mass) * chargeFactor;
    
  if (!correctEta && !correctMultiplicity)
    return dEdxSplines;
  
  Double_t corrFactorEta = 1.0;
  Double_t corrFactorMultiplicity = 1.0;
  
  if (correctEta) {
    corrFactorEta = GetEtaCorrectionFast(track, dEdxSplines);
    //TODO Alternatively take current track dEdx
    //corrFactorEta = GetEtaCorrectionFast(track, dEdx);
  }
  
  if (correctMultiplicity)
    corrFactorMultiplicity = GetMultiplicityCorrectionFast(track, dEdxSplines * corrFactorEta, fCurrentEventMultiplicity);

  return dEdxSplines * corrFactorEta * corrFactorMultiplicity;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSignal(const AliVTrack* track,
                                              AliPID::EParticleType species,
                                              ETPCdEdxSource dedxSource,
                                              Bool_t correctEta,
                                              Bool_t correctMultiplicity) const
{
  // Calculates the expected PID signal as the function of 
  // the information stored in the track, for the specified particle type 
  //  
  // At the moment, these signals are just the results of calling the 
  // Bethe-Bloch formula plus, if desired, taking into account the eta dependence
  // and the multiplicity dependence (for PbPb). 
  // This can be improved. By taking into account the number of
  // assigned clusters and/or the track dip angle, for example.  
  //
  
  if (!fUseDatabase) {
    //charge factor. BB goes with z^2, however in reality it is slightly larger (calibration, threshold effects, ...)
    // !!! Splines for light nuclei need to be normalised to this factor !!!
    const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(species),2.3);
  
    return Bethe(track->GetTPCmomentum() / AliPID::ParticleMassZ(species)) * chargeFactor;
  }
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
    
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction)) {
    // Something is wrong with the track -> Return obviously invalid value
    return -999;
  }
  
  // Charge factor already taken into account inside the following function call
  return GetExpectedSignal(track, species, dEdx, responseFunction, correctEta, correctMultiplicity);
}
  
//_________________________________________________________________________
TSpline3* AliTPCPIDResponse::GetResponseFunction( AliPID::EParticleType type,
                                                  AliTPCPIDResponse::ETPCgainScenario gainScenario ) const
{
  //get response function
  return dynamic_cast<TSpline3*>(fResponseFunctions.At(ResponseFunctionIndex(type,gainScenario)));
}

//_________________________________________________________________________
TSpline3* AliTPCPIDResponse::GetResponseFunction( const AliVTrack* track,
                               AliPID::EParticleType species,
                               ETPCdEdxSource dedxSource) const 
{
  //the splines are stored in an array, different scenarios

  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return responseFunction;
  
  return NULL;
}

//_________________________________________________________________________
void AliTPCPIDResponse::ResetSplines()
{
  //reset the array with splines
  for (Int_t i=0;i<fResponseFunctions.GetEntriesFast();i++)
  {
    fResponseFunctions.AddAt(NULL,i);
  }
}
//_________________________________________________________________________
Int_t AliTPCPIDResponse::ResponseFunctionIndex( AliPID::EParticleType species,
                                                ETPCgainScenario gainScenario ) const
{
  //get the index in fResponseFunctions given type and scenario
  return Int_t(species)+Int_t(gainScenario)*fgkNumberOfParticleSpecies;
}

//_________________________________________________________________________
void AliTPCPIDResponse::SetResponseFunction( TObject* o,
                                             AliPID::EParticleType species,
                                             ETPCgainScenario gainScenario )
{
  fResponseFunctions.AddAtAndExpand(o,ResponseFunctionIndex(species,gainScenario));
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSigma(const AliVTrack* track, 
                                             AliPID::EParticleType species,
                                             ETPCgainScenario gainScenario,
                                             Double_t dEdx,
                                             Int_t nPoints,
                                             const TSpline3* responseFunction,
                                             Bool_t correctEta,
                                             Bool_t correctMultiplicity) const 
{
  // Calculates the expected sigma of the PID signal as the function of 
  // the information stored in the track and the given parameters,
  // for the specified particle type 
  //
  
  //if (!responseFunction)
    //return 999;
    
  //TODO Check whether it makes sense to set correctMultiplicity to kTRUE while correctEta might be kFALSE
  
  // If eta correction (=> new sigma parametrisation) is requested, but no sigma map is available, print error message
  if (correctEta && !fhEtaSigmaPar1) {
    AliError("New sigma parametrisation requested, but sigma map not initialised (usually via AliPIDResponse). Old sigma parametrisation will be used!");
  }
  
  // If no sigma map is available or if no eta correction is requested (sigma maps only for corrected eta!), use the old parametrisation
  if (!fhEtaSigmaPar1 || !correctEta) {  
    if (nPoints != 0) 
      return GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, correctMultiplicity) *
               fRes0[gainScenario] * sqrt(1. + fResN2[gainScenario]/nPoints);
    else
      return GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, correctMultiplicity)*fRes0[gainScenario];
  }
    
  if (nPoints > 0) {
    // Use eta correction (+ eta-dependent sigma)
    Double_t sigmaPar1 = GetSigmaPar1Fast(track, species, dEdx, responseFunction);
    
    if (correctMultiplicity) {
      // In addition, take into account multiplicity dependence of mean and sigma of dEdx
      Double_t dEdxExpectedEtaCorrected = GetExpectedSignal(track, species, dEdx, responseFunction, kTRUE, kFALSE);
      
      // GetMultiplicityCorrection and GetMultiplicitySigmaCorrection both need the eta corrected dEdxExpected
      Double_t multiplicityCorrFactor = GetMultiplicityCorrectionFast(track, dEdxExpectedEtaCorrected, fCurrentEventMultiplicity);
      Double_t multiplicitySigmaCorrFactor = GetMultiplicitySigmaCorrectionFast(dEdxExpectedEtaCorrected, fCurrentEventMultiplicity);
      
      // multiplicityCorrFactor to correct dEdxExpected for multiplicity. In addition: Correction factor for sigma
      return (dEdxExpectedEtaCorrected * multiplicityCorrFactor) 
              * (sqrt(fSigmaPar0 * fSigmaPar0 + sigmaPar1 * sigmaPar1 / nPoints) * multiplicitySigmaCorrFactor);
    }
    else {
      return GetExpectedSignal(track, species, dEdx, responseFunction, kTRUE, kFALSE)*
             sqrt(fSigmaPar0 * fSigmaPar0 + sigmaPar1 * sigmaPar1 / nPoints);
    }
  }
  else { 
    // One should never have/take tracks with 0 dEdx clusters!
    return 999;
  }
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetExpectedSigma(const AliVTrack* track, 
                                             AliPID::EParticleType species,
                                             ETPCdEdxSource dedxSource,
                                             Bool_t correctEta,
                                             Bool_t correctMultiplicity) const 
{
  // Calculates the expected sigma of the PID signal as the function of 
  // the information stored in the track, for the specified particle type 
  // and dedx scenario
  //
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return 999; //TODO: better handling!
  
  return GetExpectedSigma(track, species, gainScenario, dEdx, nPoints, responseFunction, correctEta, correctMultiplicity);
}


//_________________________________________________________________________
Float_t AliTPCPIDResponse::GetNumberOfSigmas(const AliVTrack* track, 
                             AliPID::EParticleType species,
                             ETPCdEdxSource dedxSource,
                             Bool_t correctEta,
                             Bool_t correctMultiplicity) const
{
  //Calculates the number of sigmas of the PID signal from the expected value
  //for a given particle species in the presence of multiple gain scenarios
  //inside the TPC
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return -999; //TODO: Better handling!
    
  Double_t bethe = GetExpectedSignal(track, species, dEdx, responseFunction, correctEta, correctMultiplicity);
  Double_t sigma = GetExpectedSigma(track, species, gainScenario, dEdx, nPoints, responseFunction, correctEta, correctMultiplicity);
  // 999 will be returned by GetExpectedSigma e.g. in case of 0 dEdx clusters
  if (sigma >= 998) 
    return -999;
  else
    return (dEdx-bethe)/sigma;
}

//_________________________________________________________________________
Float_t AliTPCPIDResponse::GetSignalDelta(const AliVTrack* track,
                                          AliPID::EParticleType species,
                                          ETPCdEdxSource dedxSource,
                                          Bool_t correctEta,
                                          Bool_t correctMultiplicity,
                                          Bool_t ratio/*=kFALSE*/)const
{
  //Calculates the number of sigmas of the PID signal from the expected value
  //for a given particle species in the presence of multiple gain scenarios
  //inside the TPC

  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;

  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return -9999.; //TODO: Better handling!

  const Double_t bethe = GetExpectedSignal(track, species, dEdx, responseFunction, correctEta, correctMultiplicity);

  Double_t delta=-9999.;
  if (!ratio) delta=dEdx-bethe;
  else if (bethe>1.e-20) delta=dEdx/bethe;
  
  return delta;
}

//_________________________________________________________________________
Bool_t AliTPCPIDResponse::ResponseFunctiondEdxN( const AliVTrack* track, 
                                                 AliPID::EParticleType species,
                                                 ETPCdEdxSource dedxSource,
                                                 Double_t& dEdx,
                                                 Int_t& nPoints,
                                                 ETPCgainScenario& gainScenario,
                                                 TSpline3** responseFunction) const 
{
  // Calculates the right parameters for PID
  //   dEdx parametrization for the proper gain scenario, dEdx 
  //   and NPoints used for dEdx
  // based on the track geometry (which chambers it crosses) for the specified particle type 
  // and preferred source of dedx.
  // returns true on success
  
  
  if (dedxSource == kdEdxDefault) {
    // Fast handling for default case. In addition: Keep it simple (don't call additional functions) to
    // avoid possible bugs
    
    // GetTPCsignalTunedOnData will be non-positive, if it has not been set (i.e. in case of MC NOT tuned to data).
    // If this is the case, just take the normal signal
    dEdx = track->GetTPCsignalTunedOnData();
    if (dEdx <= 0) {
//       dEdx = track->GetTPCsignal();
      dEdx = GetTrackdEdx(track);
    }
    
    nPoints = track->GetTPCsignalN();
    gainScenario = kDefault;
    
    TObject* obj = fResponseFunctions.UncheckedAt(ResponseFunctionIndex(species,gainScenario));
    *responseFunction = dynamic_cast<TSpline3*>(obj); //TODO:maybe static cast?
  
    return kTRUE;
  }
  
  //TODO Proper handle of tuneMConData for other dEdx sources
  
  Double32_t signal[4]; //0: IROC, 1: OROC medium, 2:OROC long, 3: OROC all (def. truncation used)
  Char_t ncl[3];        //same
  Char_t nrows[3];      //same
  AliTPCdEdxInfo dEdxInfo;
  bool dEdxInfoOK = track->GetTPCdEdxInfo( dEdxInfo );
  
  if (!dEdxInfoOK && dedxSource!=kdEdxDefault)  //in one case its ok if we dont have the info
  {
    AliError("AliTPCdEdxInfo not available");
    return kFALSE;
  }

  if (dEdxInfoOK) dEdxInfo.GetTPCSignalRegionInfo(signal,ncl,nrows);

  //check if we cross a bad OROC in which case we reject
  EChamberStatus trackOROCStatus = TrackStatus(track,2);
  if (trackOROCStatus==kChamberOff || trackOROCStatus==kChamberLowGain)
  {
    return kFALSE;
  }

  switch (dedxSource)
  {
    case kdEdxOROC:
      {
        if (trackOROCStatus==kChamberInvalid) return kFALSE; //never reached OROC
        dEdx = signal[3];
        nPoints = ncl[2]+ncl[1];
        gainScenario = kOROChigh;
        break;
      }
    case kdEdxHybrid:
      {
        //if we cross a bad IROC we use OROC dedx, if we dont we use combined
        EChamberStatus status = TrackStatus(track,1);
        if (status!=kChamberHighGain)
        {
          dEdx = signal[3];
          nPoints = ncl[2]+ncl[1];
          gainScenario = kOROChigh;
        }
        else
        {
//           dEdx = track->GetTPCsignal();
          dEdx = GetTrackdEdx(track);
          nPoints = track->GetTPCsignalN();
          gainScenario = kALLhigh;
        }
        break;
      }
    default:
      {
         dEdx = 0.;
         nPoints = 0;
         gainScenario = kGainScenarioInvalid;
         return kFALSE;
      }
  }
  TObject* obj = fResponseFunctions.UncheckedAt(ResponseFunctionIndex(species,gainScenario));
  *responseFunction = dynamic_cast<TSpline3*>(obj); //TODO:maybe static cast?
  
  return kTRUE;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetEtaCorrectionFast(const AliVTrack *track, Double_t dEdxSplines) const
{
  // NOTE: For expert use only -> Non-experts are advised to use the function without the "Fast" suffix or stick to AliPIDResponse directly.
  //
  // Get eta correction for the given parameters.
  //
  
  if (!fhEtaCorr) {
    // Calling this function means to request eta correction in some way. Print error message, if no map is available!
    AliError("Eta correction requested, but map not initialised (usually via AliPIDResponse). Returning eta correction factor 1!");
    return 1.;
  }
  
  Double_t tpcSignal = dEdxSplines;
  
  if (tpcSignal < 1.)
    return 1.;
  
  Double_t tanTheta = GetTrackTanTheta(track); 
  Int_t binX = fhEtaCorr->GetXaxis()->FindFixBin(tanTheta);
  Int_t binY = fhEtaCorr->GetYaxis()->FindFixBin(1. / tpcSignal);
  
  if (binX == 0) 
    binX = 1;
  if (binX > fhEtaCorr->GetXaxis()->GetNbins())
    binX = fhEtaCorr->GetXaxis()->GetNbins();
  
  if (binY == 0)
    binY = 1;
  if (binY > fhEtaCorr->GetYaxis()->GetNbins())
    binY = fhEtaCorr->GetYaxis()->GetNbins();
  
  return fhEtaCorr->GetBinContent(binX, binY);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetEtaCorrection(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get eta correction for the given track.
  //
  
  if (!fhEtaCorr)
    return 1.;
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return 1.; 
  
  // For the eta correction, do NOT take the multiplicity corrected value of dEdx
  Double_t dEdxSplines = GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, kFALSE);
  
  //TODO Alternatively take current track dEdx
  //return GetEtaCorrectionFast(track, dEdx);
  
  return GetEtaCorrectionFast(track, dEdxSplines);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetEtaCorrectedTrackdEdx(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get eta corrected dEdx for the given track. For the correction, the expected dEdx of
  // the specified species will be used. If the species is set to AliPID::kUnknown, the
  // dEdx of the track is used instead.
  // WARNING: In the latter case, the eta correction might not be as good as if the
  // expected dEdx is used, which is the way the correction factor is designed
  // for.
  // In any case, one has to decide carefully to which expected signal one wants to
  // compare the corrected value - to the corrected or uncorrected.
  // Anyhow, a safer way of looking e.g. at the n-sigma is to call
  // the corresponding function GetNumberOfSigmas!
  //
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  // Note: In case of species == AliPID::kUnknown, the responseFunction might not be set. However, in this case
  // it is not used anyway, so this causes no trouble.
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return -1.;
  
  Double_t etaCorr = 0.;
  
  if (species < AliPID::kUnknown) {
    // For the eta correction, do NOT take the multiplicity corrected value of dEdx
    Double_t dEdxSplines = GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, kFALSE);
    etaCorr = GetEtaCorrectionFast(track, dEdxSplines);
  }
  else {
    etaCorr = GetEtaCorrectionFast(track, dEdx);
  }
    
  if (etaCorr <= 0)
    return -1.;
  
  return dEdx / etaCorr; 
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetSigmaPar1Fast(const AliVTrack *track, AliPID::EParticleType species, Double_t dEdx,
                                             const TSpline3* responseFunction) const
{
  // NOTE: For expert use only -> Non-experts are advised to use the function without the "Fast" suffix or stick to AliPIDResponse directly.
  //
  // Get parameter 1 of sigma parametrisation of TPC dEdx from the histogram for the given track.
  //
  
  if (!fhEtaSigmaPar1) {
    // Calling this function means to request new sigma parametrisation in some way. Print error message, if no map is available!
    AliError("New sigma parametrisation requested, but sigma map not initialised (usually via AliPIDResponse). Returning error value for sigma parameter1 = 999!");
    return 999;
  }
  
  // The sigma maps are created with data sets that are already eta corrected and for which the 
  // splines have been re-created. Therefore, the value for the lookup needs to be the value of
  // the splines without any additional eta correction.
  // NOTE: This is due to the method the maps are created. The track dEdx (not the expected one!)
  // is corrected to uniquely related a momemtum bin with an expected dEdx, where the expected dEdx
  // equals the track dEdx for all eta (thanks to the correction and the re-fit of the splines).
  // Consequently, looking up the uncorrected expected dEdx at a given tanTheta yields the correct
  // sigma parameter!
  // Also: It has to be the spline dEdx, since one wants to get the sigma for the assumption(!)
  // of such a particle, which by assumption then has this dEdx value
  
  // For the eta correction, do NOT take the multiplicity corrected value of dEdx
  Double_t dEdxExpected = GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, kFALSE);
  
  if (dEdxExpected < 1.)
    return 999;
  
  Double_t tanTheta = GetTrackTanTheta(track);
  Int_t binX = fhEtaSigmaPar1->GetXaxis()->FindFixBin(tanTheta);
  Int_t binY = fhEtaSigmaPar1->GetYaxis()->FindFixBin(1. / dEdxExpected);
    
  if (binX == 0) 
    binX = 1;
  if (binX > fhEtaSigmaPar1->GetXaxis()->GetNbins())
    binX = fhEtaSigmaPar1->GetXaxis()->GetNbins();
    
  if (binY == 0)
    binY = 1;
  if (binY > fhEtaSigmaPar1->GetYaxis()->GetNbins())
    binY = fhEtaSigmaPar1->GetYaxis()->GetNbins();
    
  return fhEtaSigmaPar1->GetBinContent(binX, binY);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetSigmaPar1(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get parameter 1 of sigma parametrisation of TPC dEdx from the histogram for the given track.
  //
  
  if (!fhEtaSigmaPar1)
    return 999;
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return 999; 
  
  return GetSigmaPar1Fast(track, species, dEdx, responseFunction);
}


//_________________________________________________________________________
Bool_t AliTPCPIDResponse::SetEtaCorrMap(TH2D* hMap)
{
  //
  // Load map for TPC eta correction (a copy is stored and will be deleted automatically).
  // If hMap is 0x0,the eta correction will be disabled and kFALSE is returned. 
  // If the map can be set, kTRUE is returned.
  //
  
  delete fhEtaCorr;
  
  if (!hMap) {
    fhEtaCorr = 0x0;
    
    return kFALSE;
  }
  
  fhEtaCorr = (TH2D*)(hMap->Clone());
  fhEtaCorr->SetDirectory(0);
      
  return kTRUE;
}


//_________________________________________________________________________
Bool_t AliTPCPIDResponse::SetSigmaParams(TH2D* hSigmaPar1Map, Double_t sigmaPar0)
{
  //
  // Load map for TPC sigma map (a copy is stored and will be deleted automatically):
  // Parameter 1 is stored as a 2D map (1/dEdx vs. tanTheta_local) and parameter 0 is
  // a constant. If hSigmaPar1Map is 0x0, the old sigma parametrisation will be used
  // (and sigmaPar0 is ignored!) and kFALSE is returned. 
  // If the map can be set, sigmaPar0 is also set and kTRUE will be returned.
  //
  
  delete fhEtaSigmaPar1;
  
  if (!hSigmaPar1Map) {
    fhEtaSigmaPar1 = 0x0;
    fSigmaPar0 = 0.0;
    
    return kFALSE;
  }
  
  fhEtaSigmaPar1 = (TH2D*)(hSigmaPar1Map->Clone());
  fhEtaSigmaPar1->SetDirectory(0);
  fSigmaPar0 = sigmaPar0;
  
  return kTRUE;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetTrackTanTheta(const AliVTrack *track) const
{
  // Extract the tanTheta from the information available in the AliVTrack
  
  // For ESD tracks, the local tanTheta could be used (esdTrack->GetInnerParam()->GetTgl()).
  // However, this value is not available for AODs and, thus, not for AliVTrack.
  // Fortunately, the following formula allows to approximate the local tanTheta with the 
  // global theta angle -> This is for by far most of the tracks the same, but gives at
  // maybe the percent level differences within +- 0.2 in tanTheta -> Which is still ok.
  
  /*
  const AliExternalTrackParam* innerParam = track->GetInnerParam();
  Double_t tanTheta = 0;
  if (innerParam) 
    tanTheta = innerParam->GetTgl();
  else
    tanTheta = TMath::Tan(-track->Theta() + TMath::Pi() / 2.0);
  
  // Constant in formula for B in kGauss (factor 0.1 to convert B from Tesla to kGauss),
  // pT in GeV/c (factor c*1e-9 to arrive at GeV/c) and curvature in 1/cm (factor 0.01 to get from m to cm)
  const Double_t constant = TMath::C()* 1e-9 * 0.1 * 0.01; 
  const Double_t curvature = fMagField * constant / track->Pt(); // in 1./cm
  
  Double_t averageddzdr = 0.;
  Int_t nParts = 0;

  for (Double_t r = 85; r < 245; r++) {
    Double_t sinPhiLocal = TMath::Abs(r*curvature*0.5);
    
    // Cut on |sin(phi)| as during reco
    if (TMath::Abs(sinPhiLocal) <= 0.95) {
      const Double_t phiLocal = TMath::ASin(sinPhiLocal);
      const Double_t tanPhiLocal = TMath::Tan(phiLocal);
      
      averageddzdr += tanTheta * TMath::Sqrt(1. + tanPhiLocal * tanPhiLocal); 
      nParts++;
    }
  }
  
  if (nParts > 0)
    averageddzdr /= nParts; 
  else {
    AliError("Problems during determination of dz/dr. Returning pure tanTheta as best estimate!");
    return tanTheta;
  }
  
  //printf("pT: %f\nFactor/magField(kGs)/curvature^-1: %f / %f /%f\ntanThetaGlobalFromTheta/tanTheta/Averageddzdr: %f / %f / %f\n\n",
  //          track->Pt(), constant, fMagField, 1./curvature, TMath::Tan(-track->Theta() + TMath::Pi() / 2.0), tanTheta, averageddzdr);
  
  return averageddzdr;
  */
  
  
  // Alternatively (in average, the effect is found to be negligable!):
  // Take local tanTheta from TPC inner wall, if available (currently only for ESDs available)
  //const AliExternalTrackParam* innerParam = track->GetInnerParam();
  //if (innerParam) {
  //  return innerParam->GetTgl();
  //}
  
  return TMath::Tan(-track->Theta() + TMath::Pi() / 2.0);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetMultiplicityCorrectionFast(const AliVTrack *track, Double_t dEdxExpected, Int_t multiplicity) const
{
  // NOTE: For expert use only -> Non-experts are advised to use the function without the "Fast" suffix or stick to AliPIDResponse directly.
  //
  // Calculate the multiplicity correction factor for this track for the given multiplicity.
  // The parameter dEdxExpected should take into account the eta correction already!
  
  // Multiplicity depends on pure dEdx. Therefore, correction factor depends indirectly on eta
  // => Use eta corrected value for dEdexExpected.
  
  if (dEdxExpected <= 0 || multiplicity <= 0)
    return 1.0;
  
  const Double_t dEdxExpectedInv = 1. / dEdxExpected;
  
  Double_t multCorrectionFactor = 1.0;
  
  if (!fIsNewPbPbParam) {
    Double_t relSlope = fCorrFuncMultiplicity->Eval(dEdxExpectedInv);
    
    const Double_t tanTheta = GetTrackTanTheta(track);
    relSlope += fCorrFuncMultiplicityTanTheta->Eval(tanTheta);
    
    multCorrectionFactor+= relSlope * multiplicity;
  }
  else {
    Double_t relSlope = fCorrFuncSlope->Eval(dEdxExpectedInv);
    Double_t relCurv = fCorrFuncCurv->Eval(dEdxExpectedInv);
    
    if (multiplicity <= -relSlope/(2*relCurv)) 
      multCorrectionFactor += relSlope * multiplicity + relCurv * multiplicity * multiplicity;
    else
      multCorrectionFactor -= 0.25 *relSlope * relSlope/relCurv;  
  }

  return multCorrectionFactor;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetMultiplicityCorrection(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get multiplicity correction for the given track (w.r.t. the multiplicity of the current event)
  //
  
  //TODO Should return error value, if no eta correction is enabled (needs eta correction). OR: Reset correction in case of no eta correction in PIDresponse
  
  // No correction in case of multiplicity <= 0
  if (fCurrentEventMultiplicity <= 0)
    return 1.;
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return 1.; 
  
  //TODO Does it make sense to use the multiplicity correction WITHOUT eta correction?! Are the fit parameters still valid?
  // To get the expected signal to determine the multiplicity correction, do NOT ask for the multiplicity corrected value (of course)
  Double_t dEdxExpected = GetExpectedSignal(track, species, dEdx, responseFunction, kTRUE, kFALSE);
  
  return GetMultiplicityCorrectionFast(track, dEdxExpected, fCurrentEventMultiplicity);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetMultiplicityCorrectedTrackdEdx(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get multiplicity corrected dEdx for the given track. For the correction, the expected dEdx of
  // the specified species will be used. If the species is set to AliPID::kUnknown, the
  // dEdx of the track is used instead.
  // WARNING: In the latter case, the correction might not be as good as if the
  // expected dEdx is used, which is the way the correction factor is designed
  // for.
  // In any case, one has to decide carefully to which expected signal one wants to
  // compare the corrected value - to the corrected or uncorrected.
  // Anyhow, a safer way of looking e.g. at the n-sigma is to call
  // the corresponding function GetNumberOfSigmas!
  //
  
  //TODO Should return error value, if no eta correction is enabled (needs eta correction). OR: Reset correction in case of no eta correction in PIDresponse
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  // Note: In case of species == AliPID::kUnknown, the responseFunction might not be set. However, in this case
  // it is not used anyway, so this causes no trouble.
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return -1.;
  
  
  // No correction in case of multiplicity <= 0
  if (fCurrentEventMultiplicity <= 0)
    return dEdx;
  
  Double_t multiplicityCorr = 0.;
  
  // TODO Normally, one should use the eta corrected values in BOTH of the following cases. Does it make sense to use the uncorrected dEdx values?
  if (species < AliPID::kUnknown) {
    // To get the expected signal to determine the multiplicity correction, do NOT ask for the multiplicity corrected value (of course).
    // However, one needs the eta corrected value!
    Double_t dEdxSplines = GetExpectedSignal(track, species, dEdx, responseFunction, kTRUE, kFALSE);
    multiplicityCorr = GetMultiplicityCorrectionFast(track, dEdxSplines, fCurrentEventMultiplicity);
  }
  else {
    // One needs the eta corrected value to determine the multiplicity correction factor!
    Double_t etaCorr = GetEtaCorrectionFast(track, dEdx);
    multiplicityCorr = GetMultiplicityCorrectionFast(track, dEdx * etaCorr, fCurrentEventMultiplicity);
  }
    
  if (multiplicityCorr <= 0)
    return -1.;
  
  return dEdx / multiplicityCorr; 
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetEtaAndMultiplicityCorrectedTrackdEdx(const AliVTrack *track, AliPID::EParticleType species,
                                                                    ETPCdEdxSource dedxSource) const
{
  //
  // Get multiplicity and eta corrected dEdx for the given track. For the correction,
  // the expected dEdx of the specified species will be used. If the species is set 
  // to AliPID::kUnknown, the dEdx of the track is used instead.
  // WARNING: In the latter case, the correction might not be as good as if the
  // expected dEdx is used, which is the way the correction factor is designed
  // for.
  // In any case, one has to decide carefully to which expected signal one wants to
  // compare the corrected value - to the corrected or uncorrected.
  // Anyhow, a safer way of looking e.g. at the n-sigma is to call
  // the corresponding function GetNumberOfSigmas!
  //
  
  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;
  
  // Note: In case of species == AliPID::kUnknown, the responseFunction might not be set. However, in this case
  // it is not used anyway, so this causes no trouble.
  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return -1.;
  
  Double_t multiplicityCorr = 0.;
  Double_t etaCorr = 0.;
    
  if (species < AliPID::kUnknown) {
    // To get the expected signal to determine the multiplicity correction, do NOT ask for the multiplicity corrected value (of course)
    Double_t dEdxSplines = GetExpectedSignal(track, species, dEdx, responseFunction, kFALSE, kFALSE);
    etaCorr = GetEtaCorrectionFast(track, dEdxSplines);
    multiplicityCorr = GetMultiplicityCorrectionFast(track, dEdxSplines * etaCorr, fCurrentEventMultiplicity);
  }
  else {
    etaCorr = GetEtaCorrectionFast(track, dEdx);
    multiplicityCorr = GetMultiplicityCorrectionFast(track, dEdx * etaCorr, fCurrentEventMultiplicity);
  }

  if (multiplicityCorr <= 0 || etaCorr <= 0)
    return -1.;

  return dEdx / multiplicityCorr / etaCorr;
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetMultiplicitySigmaCorrectionFast(Double_t dEdxExpected, Int_t multiplicity) const
{
  // NOTE: For expert use only -> Non-experts are advised to use the function without the "Fast" suffix or stick to AliPIDResponse directly.
  //
  // Calculate the multiplicity sigma correction factor for the corresponding expected dEdx and for the given multiplicity.
  // The parameter dEdxExpected should take into account the eta correction already!

  // Multiplicity dependence of sigma depends on the real dEdx at zero multiplicity,
  // i.e. the eta (only) corrected dEdxExpected value has to be used
  // since all maps etc. have been created for ~zero multiplicity

  if (dEdxExpected <= 0 || multiplicity <= 0)
    return 1.0;

  Double_t relSigmaSlope = fCorrFuncSigmaMultiplicity->Eval(1. / dEdxExpected);

  return (1. + relSigmaSlope * multiplicity);
}


//_________________________________________________________________________
Double_t AliTPCPIDResponse::GetMultiplicitySigmaCorrection(const AliVTrack *track, AliPID::EParticleType species, ETPCdEdxSource dedxSource) const
{
  //
  // Get multiplicity sigma correction for the given track (w.r.t. the multiplicity of the current event)
  //

  //TODO Should return error value, if no eta correction is enabled (needs eta correction). OR: Reset correction in case of no eta correction in PIDresponse

  // No correction in case of multiplicity <= 0
  if (fCurrentEventMultiplicity <= 0)
    return 1.;

  Double_t dEdx = -1;
  Int_t nPoints = -1;
  ETPCgainScenario gainScenario = kGainScenarioInvalid;
  TSpline3* responseFunction = 0x0;

  if (!ResponseFunctiondEdxN(track, species, dedxSource, dEdx, nPoints, gainScenario, &responseFunction))
    return 1.;

  //TODO Does it make sense to use the multiplicity correction WITHOUT eta correction?! Are the fit parameters still valid?
  // To get the expected signal to determine the multiplicity correction, do NOT ask for the multiplicity corrected value (of course)
  Double_t dEdxExpected = GetExpectedSignal(track, species, dEdx, responseFunction, kTRUE, kFALSE);

  return GetMultiplicitySigmaCorrectionFast(dEdxExpected, fCurrentEventMultiplicity);
}

//_____________________________________________________________________________
Double_t AliTPCPIDResponse::GetTrackdEdx(const AliVTrack* track) const
{
  // get dEdx of the track depending on the selected dEdx type
  if (fdEdxType == kdEdxInfo) {
    AliTPCdEdxInfo dEdxInfo;
    if ( track->GetTPCdEdxInfo( dEdxInfo ) ) {
      return dEdxInfo.GetWeightedMean(fdEdxChargeType, fdEdxWeightType, fIROCweight, fOROCmedWeight, fOROClongWeight);
    } else {
      AliError("Could not find dEdx info, using default signal");
    }
  }

  return track->GetTPCsignal();
}


//_________________________________________________________________________
void AliTPCPIDResponse::ResetMultiplicityCorrectionFunctions()
{
  // Default values: No correction, i.e. overall correction factor should be one
  
  // This function should always return 0.0, if multcorr disabled
  fCorrFuncMultiplicity->SetParameter(0, 0.);
  fCorrFuncMultiplicity->SetParameter(1, 0.);
  fCorrFuncMultiplicity->SetParameter(2, 0.);
  fCorrFuncMultiplicity->SetParameter(3, 0.);
  fCorrFuncMultiplicity->SetParameter(4, 0.);
    
  // This function should always return 0., if multCorr disabled
  fCorrFuncMultiplicityTanTheta->SetParameter(0, 0.);
  fCorrFuncMultiplicityTanTheta->SetParameter(1, 0.);
  fCorrFuncMultiplicityTanTheta->SetParameter(2, 0.);
  
  // This function should always return 0.0, if mutlcorr disabled
  fCorrFuncSigmaMultiplicity->SetParameter(0, 0.);
  fCorrFuncSigmaMultiplicity->SetParameter(1, 0.);
  fCorrFuncSigmaMultiplicity->SetParameter(2, 0.);
  fCorrFuncSigmaMultiplicity->SetParameter(3, 0.);
}


//_________________________________________________________________________
Bool_t AliTPCPIDResponse::sectorNumbersInOut(Double_t* trackPositionInner,
                                             Double_t* trackPositionOuter,
                                             Float_t& inphi,
                                             Float_t& outphi,
                                             Int_t& in,
                                             Int_t& out ) const
{
  //calculate the sector numbers (equivalent to IROC chamber numbers) a track crosses
  //for OROC chamber numbers add 36
  //returned angles are between (0,2pi)

  inphi = TMath::ATan2(trackPositionInner[1],trackPositionInner[0]);
  outphi = TMath::ATan2(trackPositionOuter[1], trackPositionOuter[0]);

  if (inphi<0) {inphi+=TMath::TwoPi();} //because ATan2 gives angles between -pi,pi
  if (outphi<0) {outphi+=TMath::TwoPi();} //because ATan2 gives angles between -pi,pi

  in = sectorNumber(inphi);
  out = sectorNumber(outphi);

  //for the C side (positive z) offset by 18
  if (trackPositionInner[2]>0.0)
  {
    in+=18;
    out+=18;
  }
  return kTRUE;
}


//_____________________________________________________________________________
Int_t AliTPCPIDResponse::sectorNumber(Double_t phi) const
{
  //calculate sector number
  const Float_t width=TMath::TwoPi()/18.;
  return TMath::Floor(phi/width);
}


//_____________________________________________________________________________
void AliTPCPIDResponse::Print(Option_t* /*option*/) const
{
  //Print info
  fResponseFunctions.Print();
}


//_____________________________________________________________________________
AliTPCPIDResponse::EChamberStatus AliTPCPIDResponse::TrackStatus(const AliVTrack* track, Int_t layer) const
{
  //status of the track: if it crosses any bad chambers return kChamberOff;
  //IROC:layer=1, OROC:layer=2
  if (layer<1 || layer>2) layer=1;
  Int_t in=0;
  Int_t out=0;
  Float_t inphi=0.;
  Float_t outphi=0.;
  Float_t innerRadius = (layer==1)?83.0:133.7;
  Float_t outerRadius = (layer==1)?133.5:247.7;

  /////////////////////////////////////////////////////////////////////////////
  //find out where track enters and leaves the layer.
  //
  Double_t trackPositionInner[3];
  Double_t trackPositionOuter[3];
 
  //if there is no inner param this could mean we're using the AOD track,
  //we still can extrapolate from the vertex - so use those params.
  const AliExternalTrackParam* ip = track->GetInnerParam();
  if (ip) track=dynamic_cast<const AliVTrack*>(ip);

  Bool_t trackAtInner = track->GetXYZAt(innerRadius, fMagField, trackPositionInner);
  Bool_t trackAtOuter = track->GetXYZAt(outerRadius, fMagField, trackPositionOuter);

  if (!trackAtInner)
  {
    //if we dont even enter inner radius we do nothing and return invalid
    inphi=0.0;
    outphi=0.0;
    in=0;
    out=0;
    return kChamberInvalid;
  }

  if (!trackAtOuter)
  {
    //if we don't reach the outer radius check that the apex is indeed within the outer radius and use apex position
    Bool_t haveApex = TrackApex(track, fMagField, trackPositionOuter);
    Float_t apexRadius = TMath::Sqrt(trackPositionOuter[0]*trackPositionOuter[0]+trackPositionOuter[1]*trackPositionOuter[1]);
    if ( haveApex && apexRadius<=outerRadius && apexRadius>innerRadius)
    {
      //printf("pt: %.2f, apexRadius: %.2f(%s), x: %.2f, y: %.2f\n",track->Pt(),apexRadius,(haveApex)?"OK":"BAD",trackPositionOuter[0],trackPositionOuter[1]);
    }
    else
    {
      inphi=0.0;
      outphi=0.0;
      in=0;
      out=0;
      return kChamberInvalid;
    }
  }


  if (!sectorNumbersInOut(trackPositionInner,
                          trackPositionOuter,
                          inphi,
                          outphi,
                          in,
                          out)) return kChamberInvalid;

  /////////////////////////////////////////////////////////////////////////////
  //now we have the location of the track we can check
  //if it is in a good/bad chamber
  //
  Bool_t sideA = kTRUE;
 
  if (((in/18)%2==1) && ((out/18)%2==1)) sideA=kFALSE;

  in=in%18;
  out=out%18;

  if (TMath::Abs(in-out)>9)
  {
    if (TMath::Max(in,out)==out)
    {
      Int_t tmp=out;
      out = in;
      in = tmp;
      Float_t tmpphi=outphi;
      outphi=inphi;
      inphi=tmpphi;
    }
    in-=18;
    inphi-=TMath::TwoPi();
  }
  else
  {
    if (TMath::Max(in,out)==in)
    {
      Int_t tmp=out;
      out=in;
      in=tmp;
      Float_t tmpphi=outphi;
      outphi=inphi;
      inphi=tmpphi;
    }
  }

  Float_t trackLengthInBad = 0.;
  Float_t trackLengthInLowGain = 0.;
  Float_t trackLengthTotal = TMath::Abs(outphi-inphi);
  Float_t lengthFractionInBadSectors = 0.;

  const Float_t sectorWidth = TMath::TwoPi()/18.;  
 
  for (Int_t i=in; i<=out; i++)
  {
    int j=i;
    if (i<0) j+=18;    //correct for the negative values
    if (!sideA) j+=18; //move to the correct side
   
    Float_t deltaPhi = 0.;
    Float_t phiEdge=sectorWidth*i;
    if (inphi>phiEdge) {deltaPhi=phiEdge+sectorWidth-inphi;}
    else if ((outphi>=phiEdge) && (outphi<(phiEdge+sectorWidth))) {deltaPhi=outphi-phiEdge;}
    else {deltaPhi=sectorWidth;}
   
    Float_t v = fVoltageMap[(layer==1)?(j):(j+36)];
    if (v<=fBadIROCthreshhold)
    {
      trackLengthInBad+=deltaPhi;
      lengthFractionInBadSectors=1.;
    }
    if (v<=fLowGainIROCthreshold && v>fBadIROCthreshhold)
    {
      trackLengthInLowGain+=deltaPhi;
      lengthFractionInBadSectors=1.;
    }
  }

  //for now low gain and bad (off) chambers are treated equally
  if (trackLengthTotal>0)
    lengthFractionInBadSectors = (trackLengthInLowGain+trackLengthInBad)/trackLengthTotal;

  //printf("### side: %s, pt: %.2f, pz: %.2f, in: %i, out: %i, phiIN: %.2f, phiOUT: %.2f, rIN: %.2f, rOUT: %.2f\n",(sideA)?"A":"C",track->Pt(),track->Pz(),in,out,inphi,outphi,innerRadius,outerRadius);
 
  if (lengthFractionInBadSectors>fMaxBadLengthFraction)
  {
    //printf("%%%%%%%% %s kChamberLowGain\n",(layer==1)?"IROC":"OROC");
    return kChamberLowGain;
  }
 
  return kChamberHighGain;
}


//_____________________________________________________________________________
Float_t AliTPCPIDResponse::MaxClusterRadius(const AliVTrack* track) const
{
  //return the radius of the outermost padrow containing a cluster in TPC
  //for the track
  const TBits* clusterMap=track->GetTPCClusterMapPtr();
  if (!clusterMap) return 0.;

  //from AliTPCParam, radius of first IROC row
  const Float_t rfirstIROC = 8.52249984741210938e+01;
  const Float_t padrowHeightIROC = 0.75;
  const Float_t rfirstOROC0 = 1.35100006103515625e+02;
  const Float_t padrowHeightOROC0 = 1.0;
  const Float_t rfirstOROC1 = 1.99350006103515625e+02;
  const Float_t padrowHeightOROC1 = 1.5;

  Int_t maxPadRow=160;
  while ((--maxPadRow)>0 && !clusterMap->TestBitNumber(maxPadRow)){}
  if (maxPadRow>126) return rfirstOROC1+(maxPadRow-126-1)*padrowHeightOROC1;
  if (maxPadRow>62) return rfirstOROC0+(maxPadRow-62-1)*padrowHeightOROC0;
  if (maxPadRow>0) return rfirstIROC+(maxPadRow-1)*padrowHeightIROC;
  return 0.0;
}


//_____________________________________________________________________________
Bool_t AliTPCPIDResponse::TrackApex(const AliVTrack* track, Float_t magField, Double_t position[3]) const
{
  //calculate the coordinates of the apex of the track
  Double_t x[3];
  track->GetXYZ(x);
  Double_t p[3];
  track->GetPxPyPz(p);
  Double_t r = 1./track->OneOverPt()/0.0299792458/magField; //signed - will determine the direction of b
  //printf("b: %.2f, x:%.2f, y:%.2f, pt: %.2f, px:%.2f, py%.2f, r: %.2f\n",magField, x[0],x[1],track->Pt(), p[0],p[1],r);
  //find orthogonal vector (Gram-Schmidt)
  Double_t alpha = (p[0]*x[0] + p[1]*x[1])/(p[0]*p[0] + p[1]*p[1]);
  Double_t b[2];
  b[0]=x[0]-alpha*p[0];
  b[1]=x[1]-alpha*p[1];
 
  Double_t norm = TMath::Sqrt(b[0]*b[0]+b[1]*b[1]);
  if (TMath::AreEqualAbs(norm,0.0,1e-10)) return kFALSE;
  b[0]/=norm;
  b[1]/=norm;
  b[0]*=r;
  b[1]*=r;
  b[0]+=x[0];
  b[1]+=x[1];
  //printf("center: x:%.2f, y:%.2f\n",b[0],b[1]);
 
  norm = TMath::Sqrt(b[0]*b[0]+b[1]*b[1]);
  if (TMath::AreEqualAbs(norm,0.0,1e-10)) return kFALSE;
 
  position[0]=b[0]+b[0]*TMath::Abs(r)/norm;
  position[1]=b[1]+b[1]*TMath::Abs(r)/norm;
  position[2]=0.;
  return kTRUE;
}

Double_t AliTPCPIDResponse::EvaldEdxSpline(Double_t bg,Int_t entry){
  //
  // Evaluate the dEdx response for given entry
  //
  TSpline * spline = (TSpline*)fSplineArray.At(entry);
  if (spline) return spline->Eval(bg);
  return 0;
}


Bool_t   AliTPCPIDResponse::RegisterSpline(const char * name, Int_t index){
  //
  // register spline to be used for drawing comparisons
  // 
  TFile * fTPCBB = TFile::Open("$ALICE_PHYSICS/OADB/COMMON/PID/data/TPCPIDResponse.root");
  TObjArray  *arrayTPCPID= (TObjArray*)  fTPCBB->Get("TPCPIDResponse");
  if (fSplineArray.GetEntriesFast()<index) fSplineArray.Expand(index*2);
  TSpline3 *spline=0;
  if (arrayTPCPID){
    spline = (TSpline3*)arrayTPCPID->FindObject(name);
    if (spline) fSplineArray.AddAt(spline->Clone(),index);    
  }
  delete arrayTPCPID;
  delete fTPCBB;
  return (spline!=0);
}

//_____________________________________________________________________________
Bool_t AliTPCPIDResponse::InitFromOADB(const Int_t run, const Int_t pass, TString passName,
                                       const char* oadbFile/*="$ALICE_PHYSICS/OADB/COMMON/PID/data/TPCPIDResponseOADB.root"*/,
                                       Bool_t initMultiplicityCorrection/*=kTRUE*/)
{
  //
  //
  //

  AliInfo( "----------------------| Initialisation TPC PID Response from OADB |----------------------");
  AliInfoF("----------------------| Run: %d, pass: %d - %-16s |----------------------", run, pass, passName.Data());
  if (!fOADBContainer) {
    fOADBContainer = new AliOADBContainer("TPCSplines");
    fOADBContainer->InitFromFile(oadbFile,"TPCSplines");
  }

  const TString spass=TString::Format("%d", pass);
  const Int_t passOrig=pass;
  Int_t passIter=passOrig;

  fRecoPassNameUsed="";

  const TObjArray *arr=0x0;
  // first try to find an object using the full pass name
  if (!passName.IsNull()) {
    AliInfoF("Trying to load splines for specific pass name %s.", passName.Data());
    arr = dynamic_cast<TObjArray *>(fOADBContainer->GetObject(run, "", passName.Data()));
    if (arr) {
      AliInfoF("Dedicated splines for '%s' found.", passName.Data());
      fRecoPassNameUsed=passName;
    }
    else {
      AliInfoF("No dedicated splines found for '%s', check numerical pass %s.", passName.Data(), spass.Data());
    }
  }
  // ... then fall back to numerical pass number
  if (!arr) {
    AliInfoF("Trying to load splines for specific reconstruction pass %s.", spass.Data());
    arr = dynamic_cast<TObjArray *>(fOADBContainer->GetObject(run, "", spass.Data()));
    fRecoPassNameUsed=spass;
  }
  // ... then try with previous passes and issue a warning
  while (!arr && --passIter > 0) {
    TString passCurrent=TString::Format("%d", passIter);
    arr=dynamic_cast<TObjArray*>(fOADBContainer->GetObject(run,"",passCurrent.Data()));
  }

  if (!arr) {
    AliError ("***** Risk for unreliable TPC PID detected:                      ********");
    AliErrorF("      could not find a valid OADB object for run %d pass %d (%s)", run, pass, passName.Data());
    AliError ("      Most probably this is because the PID response for this pass is not available, yet");
    AliError ("      Please also check https://twiki.cern.ch/twiki/bin/view/ALICE/TPCSplines");
    fRecoPassNameUsed="";
    return kFALSE;
  }

  if (passIter<passOrig) {
    AliWarning(     "***** Risk for unreliable TPC PID detected:                      ********");
    AliWarning(Form("      Using splines from a previous pass: %d<%d", passIter, passOrig));
    AliWarning(     "      Most probably this is because the PID response for this pass has not been extracted, yet");
    AliWarning(     "      Please also check https://twiki.cern.ch/twiki/bin/view/ALICE/TPCSplines");
    fRecoPassNameUsed=TString::Format("%d", passIter);
  }

  //===| Set up of splines |====================================================
  // clear response functions and reset spline usage
  fResponseFunctions.Clear();
  SetUseDatabase(kFALSE);

  const TObjArray *arrSplines = static_cast<TObjArray*>(arr->FindObject("Splines"));
  if (!arrSplines) {
    AliError("***** Risk for unreliable TPC PID detected:                      ********");
    AliError(Form("      could not find array of splines for run %d", run));
    AliError(     "      This should not happen, plese report");
    return kFALSE;
  }
  SetSplinesFromArray(arrSplines);

  //===| Set up multiplicity correction |=======================================
  if (initMultiplicityCorrection) {
    const TObject *multiplicityCorrection=arr->FindObject("MultiplicityCorrection");
    if (multiplicityCorrection) {
      const TString multiplicityData(multiplicityCorrection->GetTitle());
      const Bool_t res=SetMultiplicityCorrectionFromString(multiplicityData);
      if (!res) {
        AliError("Problem setting up multiplicity correction for TPC PID");
      }
    }
  } else {
    AliInfo("Multiplicity correction explicitly disabled");
  }

  //===| Set up of dEdx type |==================================================
  const TNamed *dEdxType=static_cast<TNamed*>(arr->FindObject("dEdxType"));
  if (dEdxType) {
    const TString dEdxTypeSet(dEdxType->GetTitle());
    SetdEdxTypeFromString(dEdxTypeSet);
  }

  //===| resolution parametrisation |===========================================
  const TNamed *dEdxResolution=static_cast<TNamed*>(arr->FindObject("dEdxResolution"));
  if (dEdxResolution) {
    const TString dEdxResolutionString(dEdxResolution->GetTitle());
    SetdEdxResolutionFromString(dEdxResolutionString);
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t AliTPCPIDResponse::SetSplinesFromArray(const TObjArray* arrSplines)
{
  // Set up internal spline array from order array of splines 'arrSplines'
  // arrSplines is assumes to have the splines for the single particles inc
  // the numbered position as given by AliPID::EParticleType

  //---| get default splines for missing species |------------------------------
  TObject *protonSpline = arrSplines->At(AliPID::kProton  );
  TObject *pionSpline   = arrSplines->At(AliPID::kPion    );
//   TObject *allSpline    = arrSplines->At(AliPID::kSPECIESC);

  //---| set up the spline array |----------------------------------------------
  for (Int_t ispecie=0; ispecie<AliPID::kSPECIESC; ++ispecie) {
    TSpline3 *responseFunction = static_cast<TSpline3*>(arrSplines->At(ispecie));

    //---| spline replacement |-------------------------------------------------
    //     usually no splines are extracted for muons and light nuclei
    //     in case there is no dedicated spline, use the
    //     * pion spline for muons and the
    //     * proton spline for light nuclei
    if (!responseFunction) {
      if (ispecie==Int_t(AliPID::kMuon)) {
        responseFunction=static_cast<TSpline3*>(pionSpline);
      } else if (ispecie>Int_t(AliPID::kProton)) {
        responseFunction=static_cast<TSpline3*>(protonSpline);
      }
    }

    if (!responseFunction) {
      AliError("***** Risk for unreliable TPC PID detected:                      ********");
      AliError(Form("      No spline found for %s, this should not happen, please report", AliPID::ParticleName(ispecie) ));
      continue;
    }
    SetUseDatabase(kTRUE);
    SetResponseFunction((AliPID::EParticleType)ispecie, responseFunction);
    AliInfo(Form("Adding spline: %d - %s (MD5(spline) = %s)",ispecie,responseFunction->GetName(),
                 GetChecksum(responseFunction).Data()));

  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t AliTPCPIDResponse::SetMultiplicityCorrectionFromString(const TString& multiplicityData)
{
  //
  //
  //

  const TObjArray *arrParameters=0x0;
  if (!(arrParameters=GetMultiplicityCorrectionArrayFromString(multiplicityData))) {
    return kFALSE;
  }
  
  if (((TObjString*)arrParameters->At(arrParameters->GetLast()))->String() == "1") {
    fIsNewPbPbParam = kTRUE;
  }

  TString log("Setting multiplicity correction parameters for mean; tan theta; sigma: ");
  log.Append(multiplicityData);
  AliInfo(log.Data());
  
  const TObjArray *arrPar1 = static_cast<TObjArray*>(arrParameters->At(0));
  for (Int_t ipar=0; ipar<arrPar1->GetEntriesFast(); ++ipar) {
    if (fIsNewPbPbParam) 
      SetParameterMultSlope(ipar, static_cast<TObjString*>(arrPar1->At(ipar))->String().Atof());
    else
      SetParameterMultiplicityCorrection(ipar, static_cast<TObjString*>(arrPar1->At(ipar))->String().Atof());
  }

  const TObjArray *arrPar2 = static_cast<TObjArray*>(arrParameters->At(1));
  for (Int_t ipar=0; ipar<arrPar2->GetEntriesFast(); ++ipar) {
    if (fIsNewPbPbParam) 
      SetParameterMultCurv(ipar, static_cast<TObjString*>(arrPar2->At(ipar))->String().Atof());
    else    
      SetParameterMultiplicityCorrectionTanTheta(ipar, static_cast<TObjString*>(arrPar2->At(ipar))->String().Atof());
  }

  const TObjArray *arrPar3 = static_cast<TObjArray*>(arrParameters->At(2));
  for (Int_t ipar=0; ipar<arrPar3->GetEntriesFast(); ++ipar) {
    SetParameterMultiplicitySigmaCorrection(ipar, static_cast<TObjString*>(arrPar3->At(ipar))->String().Atof());
  }

  delete arrParameters;
  return kTRUE;
}

//______________________________________________________________________________
TObjArray* AliTPCPIDResponse::GetMultiplicityCorrectionArrayFromString(const TString& corrections)
{
  // the corrections string is supposed to be in the format
  // parMC_0,parMC_1,parMC_2,parMC_3, parMC_4; parMCTT_0,parMCTT_1,parMCTT_2; parMSC_0,parMSC_1,parMSC_2, parMSC_3
  // where parMC are the parameters for AliTPCPIDResponse::SetParameterMultiplicityCorrection
  // parMCTT are the parameters for AliTPCPIDResponse::SetParameterMultiplicityCorrectionTanTheta
  // parMSC are the parameters for AliTPCPIDResponse::SetParameterMultiplicitySigmaCorrection
  
  //For the new PbPb parametrization it should be in the following format: parMC_0,parMC1;parMCC_0,parMCC_1;parMSC_0,parMSC_1,parMSC_2, parMSC_3 where parMC_i are the parameters for the Slope parametrization and parMCC_i the parameters for the curvature parametrization

  const Int_t nSets=3;
  const Int_t nPars_0[nSets]={5,3,4};
  const Int_t nPars_1[nSets]={2,2,4};

  AliTPCPIDResponse temp;
  TObjArray *arrCorrectionSets = corrections.Tokenize(";");
  if (arrCorrectionSets->GetEntriesFast() != nSets){
    temp.Error("AliTPCPIDResponse::CheckMultiplicityCorrectionString","Number of parameter sets not equal to %d. Please read documentation of this function", nSets);
    delete arrCorrectionSets;
    return 0x0;
  }
  
  for (Int_t iset=0 ;iset<nSets; ++iset) {
    TObjString *string=(TObjString*)arrCorrectionSets->RemoveAt(iset);
    TObjArray *arrTmp=string->String().Tokenize(",");
    delete string;
    string=0x0;

    if (arrTmp->GetEntriesFast() != nPars_0[iset] && arrTmp->GetEntriesFast() != nPars_1[iset]){
      temp.Error("AliTPCPIDResponse::CheckMultiplicityCorrectionString","Number of parameters in set %d not equal to %d or %d. Please read documentation of this function", iset, nPars_0[iset], nPars_1[iset]);
      delete arrCorrectionSets;
      delete arrTmp;
      return 0x0;
    }

    arrCorrectionSets->AddAt(arrTmp, iset);
  }
  
  Bool_t matchesParSet = kTRUE;
  for (Int_t iset=0;iset<nSets;++iset) 
    matchesParSet = matchesParSet && ((TObjArray*)arrCorrectionSets->At(iset))->GetEntriesFast() == nPars_0[iset];
  
  if (matchesParSet)
    arrCorrectionSets->AddLast(new TObjString("0"));
  else {
    matchesParSet = kTRUE;
    for (Int_t iset=0;iset<nSets;++iset) {
      matchesParSet = matchesParSet && ((TObjArray*)arrCorrectionSets->At(iset))->GetEntriesFast() == nPars_1[iset];
    }
    if (matchesParSet)
      arrCorrectionSets->AddLast(new TObjString("1"));
  }

  return arrCorrectionSets;
}

//______________________________________________________________________________
Bool_t AliTPCPIDResponse::SetdEdxTypeFromString(const TString& dEdxTypeSet)
{
  // Set up the dEdx usage parsing dEdxTypeSet
  // 6 comma separated values are expected.
  //
  // The format assumes is:
  // 0: dEdxType           (AliTPCPIDResponse::ETPCdEdxType)
  // 1: dEdxChargeType     (qTot, qMax)
  // 2: dEdxWeightType     (0-measured pid clusters, 1-measured pid clusters + subThreshold clusters)
  // 3: dEdxIROCweight     (specific weight for IROC)
  // 4: dEdxOROCmedWeight  (specific weight for OROC medium pads)
  // 5: dEdxOROClongWeight (specific weight for OROC long pads)
  TObjArray *arrParams = dEdxTypeSet.Tokenize(",");

  Bool_t retVal=kFALSE;

  if (arrParams->GetEntriesFast() == 6) {
    const Int_t    dEdxType           = ((TObjString*)arrParams->At(0))->String().Atoi();
    const Int_t    dEdxChargeType     = ((TObjString*)arrParams->At(1))->String().Atoi();
    const Int_t    dEdxWeightType     = ((TObjString*)arrParams->At(2))->String().Atoi();
    const Double_t dEdxIROCweight     = ((TObjString*)arrParams->At(3))->String().Atof();
    const Double_t dEdxOROCmedWeight  = ((TObjString*)arrParams->At(4))->String().Atof();
    const Double_t dEdxOROClongWeight = ((TObjString*)arrParams->At(5))->String().Atof();
    SetdEdxType((AliTPCPIDResponse::ETPCdEdxType)dEdxType, dEdxChargeType, dEdxWeightType, dEdxIROCweight, dEdxOROCmedWeight, dEdxOROClongWeight);
    AliInfo(TString::Format("Setting custom TPC dEdxType: %d, %d, %d, %.2f, %.2f, %.2f",
                            dEdxType, dEdxChargeType, dEdxWeightType, dEdxIROCweight, dEdxOROCmedWeight, dEdxOROClongWeight));
    retVal=kTRUE;
  } else {
    AliError("Wrong number of parameters for custom TPC dEdxType");
  }
  delete arrParams;

  return retVal;
}

//______________________________________________________________________________
Bool_t AliTPCPIDResponse::SetdEdxResolutionFromString(const TString& dEdxResolutionString)
{
  // Set up the dEdx resolution parsing dEdxResolutionString
  // 2 comma separated values are expected to describe the relative resolution
  // if resN2=0 no dependence on the number of clusters will be described
  // sigma_rel = res0 * sqrt(1+resN2/npoint)
  //
  // The format assumes is:
  // 0: res0
  // 1: resN2
  TObjArray *arrParams = dEdxResolutionString.Tokenize(",");

  Bool_t retVal=kFALSE;

  if (arrParams->GetEntriesFast() == 2) {
    const Float_t    res0  = ((TObjString*)arrParams->At(0))->String().Atoi();
    const Float_t    resN2 = ((TObjString*)arrParams->At(1))->String().Atoi();
    AliInfo(TString::Format("Setting custom TPC dEdxResolution: %.2f, %.2f",
                            fRes0[0], fResN2[0]));
    fRes0 [0]=res0;
    fResN2[0]=resN2;
    retVal=kTRUE;
  } else {
    AliError("Wrong number of parameters for custom TPC dEdxResolution");
  }
  delete arrParams;

  return retVal;
}

//______________________________________________________________________________
TString AliTPCPIDResponse::GetChecksum(const TObject* obj)
{
  // Return the checksum for an object obj (tested to work properly at least for histograms and TSplines).

  TString fileName = Form("tempChecksum.C"); // File name must be fixed for data type "TSpline3", since the file name will end up in the file content!

  // For parallel processing, a unique file pathname is required. Uniqueness can be guaranteed by using a unique directory name
  UInt_t index = 0;
  TString uniquePathName = Form("tempChecksum_%u", index);

  // To get a unique path name, increase the index until no directory
  // of such a name exists.
  // NOTE: gSystem->AccessPathName(...) returns kTRUE, if the access FAILED!
  while (!gSystem->AccessPathName(uniquePathName.Data()))
    uniquePathName = Form("tempChecksum_%u", ++index);

  AliTPCPIDResponse temp;
  if (gSystem->mkdir(uniquePathName.Data()) < 0) {
    temp.Error("AliTPCPIDResponse::GetChecksum","Could not create temporary directory to store temp file for checksum determination!");
    return "ERROR";
  }

  TString option = "";

  // Save object as a macro, which will be deleted immediately after the checksum has been computed
  // (does not work for desired data types if saved as *.root for some reason) - one only wants to compare the content, not
  // the modification time etc. ...
  if (dynamic_cast<const TH1*>(obj)) {
    option = "colz"; // Histos need this option, since w/o this option, a counter is added to the filename
  }


  // SaveAs must be called with the fixed fileName only, since the first argument goes into the file content
  // for some object types. Thus, change the directory, save the file and then go back
  TString oldDir = gSystem->pwd();
  gSystem->cd(uniquePathName.Data());
  obj->SaveAs(fileName.Data(), option.Data());
  gSystem->cd(oldDir.Data());

  // Use the file to calculate the MD5 checksum
  TMD5* md5 = TMD5::FileChecksum(Form("%s/%s", uniquePathName.Data(), fileName.Data()));
  TString checksum = md5->AsString();

  // Clean up
  delete md5;
  gSystem->Exec(Form("rm -rf %s", uniquePathName.Data()));

  return checksum;
}
