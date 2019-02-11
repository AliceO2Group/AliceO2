/**************************************************************************
 * Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
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

/* $Id: */

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to store parameters of ITS response functions           //
// Origin: F.Prino, Torino, prino@to.infn.it                     //
// Modified by: Y. Corrales Morales                              //
//              Torino, corrales@to.infn.it                      //
//                                                               //
///////////////////////////////////////////////////////////////////

#include <TMath.h>
#include "AliITSPidParams.h"

ClassImp(AliITSPidParams)

//______________________________________________________________________
AliITSPidParams::AliITSPidParams(Bool_t isMC):
TNamed("default",""),
  fSDDElecMPV(0),
  fSDDElecLandauWidth(0),
  fSDDElecGaussWidth(0),
  fSSDElecMPV(0),
  fSSDElecLandauWidth(0),
  fSSDElecGaussWidth(0),
  fSDDPionMPV(0),
  fSDDPionLandauWidth(0),
  fSDDPionGaussWidth(0),
  fSSDPionMPV(0),
  fSSDPionLandauWidth(0),
  fSSDPionGaussWidth(0),
  fSDDKaonMPV(0),
  fSDDKaonLandauWidth(0),
  fSDDKaonGaussWidth(0),
  fSSDKaonMPV(0),
  fSSDKaonLandauWidth(0),
  fSSDKaonGaussWidth(0),
  fSDDProtMPV(0),
  fSDDProtLandauWidth(0),
  fSDDProtGaussWidth(0),
  fSSDProtMPV(0),
  fSSDProtLandauWidth(0),
  fSSDProtGaussWidth(0)
{
  // default constructor
  if (isMC) InitMC();
  else InitData();
}
//______________________________________________________________________
AliITSPidParams::AliITSPidParams(Char_t * name, Bool_t isMC):
  TNamed(name,""),
  fSDDElecMPV(0),
  fSDDElecLandauWidth(0),
  fSDDElecGaussWidth(0),
  fSSDElecMPV(0),
  fSSDElecLandauWidth(0),
  fSSDElecGaussWidth(0),
  fSDDPionMPV(0),
  fSDDPionLandauWidth(0),
  fSDDPionGaussWidth(0),
  fSSDPionMPV(0),
  fSSDPionLandauWidth(0),
  fSSDPionGaussWidth(0),
  fSDDKaonMPV(0),
  fSDDKaonLandauWidth(0),
  fSDDKaonGaussWidth(0),
  fSSDKaonMPV(0),
  fSSDKaonLandauWidth(0),
  fSSDKaonGaussWidth(0),
  fSDDProtMPV(0),
  fSDDProtLandauWidth(0),
  fSDDProtGaussWidth(0),
  fSSDProtMPV(0),
  fSSDProtLandauWidth(0),
  fSSDProtGaussWidth(0)
{
  // standard constructor
  if (isMC) InitMC();
  else InitData();
}
//______________________________________________________________________
AliITSPidParams::~AliITSPidParams(){

  if(fSDDElecMPV) delete fSDDElecMPV;
  if(fSDDElecLandauWidth) delete fSDDElecLandauWidth;
  if(fSDDElecGaussWidth) delete fSDDElecGaussWidth;

  if(fSSDElecMPV) delete fSSDElecMPV;
  if(fSSDElecLandauWidth) delete fSSDElecLandauWidth;
  if(fSSDElecGaussWidth) delete fSSDElecGaussWidth;

  if(fSDDPionMPV) delete fSDDPionMPV;
  if(fSDDPionLandauWidth) delete fSDDPionLandauWidth;
  if(fSDDPionGaussWidth) delete fSDDPionGaussWidth;

  if(fSSDPionMPV) delete fSSDPionMPV;
  if(fSSDPionLandauWidth) delete fSSDPionLandauWidth;
  if(fSSDPionGaussWidth) delete fSSDPionGaussWidth;

  if(fSDDKaonMPV) delete fSDDKaonMPV;
  if(fSDDKaonLandauWidth) delete fSDDKaonLandauWidth;
  if(fSDDKaonGaussWidth) delete fSDDKaonGaussWidth;

  if(fSSDKaonMPV) delete fSSDKaonMPV;
  if(fSSDKaonLandauWidth) delete fSSDKaonLandauWidth;
  if(fSSDKaonGaussWidth) delete fSSDKaonGaussWidth;

  if(fSDDProtMPV) delete fSDDProtMPV;
  if(fSDDProtLandauWidth) delete fSDDProtLandauWidth;
  if(fSDDProtGaussWidth) delete fSDDProtGaussWidth;

  if(fSSDProtMPV) delete fSSDProtMPV;
  if(fSSDProtLandauWidth) delete fSSDProtLandauWidth;
  if(fSSDProtGaussWidth) delete fSSDProtGaussWidth;
}

//______________________________________________________________________
void AliITSPidParams::InitMC(){
  // initialize TFormulas to Monte Carlo values (=p-p simulations PYTHIA+GEANT)
  // parameter values from LHC10d1
  // MPV BetheBloch parameters;

  //sdd MC electrons parameters
  Double_t fSDDElecMPVParams[2] = {  0.2507, 84.64};
  Double_t fSDDElecLWVParams[2] = { 0.02969, 6.634};
  Double_t fSDDElecGWVParams[2] = { -0.4338, 8.225};

  //ssd MC electrons parameters
  Double_t fSSDElecMPVParams[2] = {  0.2762, 86.92};
  Double_t fSSDElecLWVParams[2] = { 0.06276, 6.333};
  Double_t fSSDElecGWVParams[2] = {  0.1672, 6.519};

  // electrons
  if(fSDDElecMPV) delete fSDDElecMPV;
  fSDDElecMPV=new TFormula("fSDDElecMPV","[0]*x+[1]");
  fSDDElecMPV->SetParameters(fSDDElecMPVParams);

  if(fSDDElecLandauWidth) delete fSDDElecLandauWidth;
  fSDDElecLandauWidth=new TFormula("fSDDElecLandauWidth","[0]*x+[1]");
  fSDDElecLandauWidth->SetParameters(fSDDElecLWVParams);

  if(fSDDElecGaussWidth) delete fSDDElecGaussWidth;
  fSDDElecGaussWidth=new TFormula("fSDDElecGaussWidth","[0]*x+[1]");
  fSDDElecGaussWidth->SetParameters(fSDDElecGWVParams);

  if(fSSDElecMPV) delete fSSDElecMPV;
  fSSDElecMPV=new TFormula("fSSDElecMPV","[0]*x+[1]");
  fSSDElecMPV->SetParameters(fSSDElecMPVParams);

  if(fSSDElecLandauWidth) delete fSSDElecLandauWidth;
  fSSDElecLandauWidth=new TFormula("fSSDElecLandauWidth","[0]*x+[1]");
  fSSDElecLandauWidth->SetParameters(fSSDElecLWVParams);

  if(fSSDElecGaussWidth) delete fSSDElecGaussWidth;
  fSSDElecGaussWidth=new TFormula("fSSDElecGaussWidth","[0]*x+[1]");
  fSSDElecGaussWidth->SetParameters(fSSDElecGWVParams);

  //sdd MC hadrons parameters
  //pion
  Double_t fSDDPionMPVParams[3] = { 1.418,  3.653, 76.44};
  Double_t fSDDPionLWVParams[3] = {0.1932,  0.297, 6.392};
  Double_t fSDDPionGWVParams[3] = {0.2163, 0.7689, 6.774};
  //kaon
  Double_t fSDDKaonMPVParams[3] = { 8.44, -19.04, 81.33};
  Double_t fSDDKaonLWVParams[2] = {1.274,  6.344};
  Double_t fSDDKaonGWVParams[3] = {3.345,  2.411, 5.007};
  //proton
  Double_t fSDDProtMPVParams[3] = {-12.73, -151.4, 147.4};
  Double_t fSDDProtLWVParams[2] = { 5.858, 5.397};
  Double_t fSDDProtGWVParams[3] = { 9.169,  1.985, 5.595};

  //ssd MC hadrons parameters
  //pion
  Double_t fSSDPionMPVParams[3] = { 1.502,  4.336, 78.73};
  Double_t fSSDPionLWVParams[3] = {0.2486, 0.4315, 6.138};
  Double_t fSSDPionGWVParams[3] = {0.2253, 0.9694, 5.248};
  //kaon
  Double_t fSSDKaonMPVParams[3] = {13.72, -6.747, 78.12};
  Double_t fSSDKaonLWVParams[2] = {1.558,  5.753};
  Double_t fSSDKaonGWVParams[3] = { 4.61,  5.838, 1.799};
  //proton
  Double_t fSSDProtMPVParams[3] = {13.16, -97.03, 122.6};
  Double_t fSSDProtLWVParams[2] = {6.188,  5.177};
  Double_t fSSDProtGWVParams[3] = {18.88,  23.06, -7.24};


  // pions
  if(fSDDPionMPV) delete fSDDPionMPV;
  fSDDPionMPV=new TFormula("fSDDPionMPV","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSDDPionMPV->SetParameters(fSDDPionMPVParams);

  if(fSDDPionLandauWidth) delete fSDDPionLandauWidth;
  fSDDPionLandauWidth=new TFormula("fSDDPionLandauWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDPionLandauWidth->SetParameters(fSDDPionLWVParams);

  if(fSDDPionGaussWidth) delete fSDDPionGaussWidth;
  fSDDPionGaussWidth=new TFormula("fSDDPionGaussWidth","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSDDPionGaussWidth->SetParameters(fSDDPionGWVParams);

  if(fSSDPionMPV) delete fSSDPionMPV;
  fSSDPionMPV=new TFormula("fSSDPionMPV","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSSDPionMPV->SetParameters(fSSDPionMPVParams);

  if(fSSDPionLandauWidth) delete fSSDPionLandauWidth;
  fSSDPionLandauWidth=new TFormula("fSSDPionLandauWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDPionLandauWidth->SetParameters(fSSDPionLWVParams);

  if(fSSDPionGaussWidth) delete fSSDPionGaussWidth;
  fSSDPionGaussWidth=new TFormula("fSSDPionGaussWidth","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSSDPionGaussWidth->SetParameters(fSSDPionGWVParams);

  // kaons
  if(fSDDKaonMPV) delete fSDDKaonMPV;
  fSDDKaonMPV=new TFormula("fSDDKaonMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDKaonMPV->SetParameters(fSDDKaonMPVParams);

  if(fSDDKaonLandauWidth) delete fSDDKaonLandauWidth;
  fSDDKaonLandauWidth=new TFormula("fSDDKaonLandauWidth","[0]/(x*x)+[1]");
  fSDDKaonLandauWidth->SetParameters(fSDDKaonLWVParams);

  if(fSDDKaonGaussWidth) delete fSDDKaonGaussWidth;
  fSDDKaonGaussWidth=new TFormula("fSDDKaonGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDKaonGaussWidth->SetParameters(fSDDKaonGWVParams);

  if(fSSDKaonMPV) delete fSSDKaonMPV;
  fSSDKaonMPV=new TFormula("fSSDKaonMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDKaonMPV->SetParameters(fSSDKaonMPVParams);

  if(fSSDKaonLandauWidth) delete fSSDKaonLandauWidth;
  fSSDKaonLandauWidth=new TFormula("fSSDKaonLandauWidth","[0]/(x*x)+[1]");
  fSSDKaonLandauWidth->SetParameters(fSSDKaonLWVParams);

  if(fSSDKaonGaussWidth) delete fSSDKaonGaussWidth;
  fSSDKaonGaussWidth=new TFormula("fSSDKaonGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDKaonGaussWidth->SetParameters(fSSDKaonGWVParams);

  // protons
  if(fSDDProtMPV) delete fSDDProtMPV;
  fSDDProtMPV=new TFormula("fSDDProtMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDProtMPV->SetParameters(fSDDProtMPVParams);

  if(fSDDProtLandauWidth) delete fSDDProtLandauWidth;
  fSDDProtLandauWidth=new TFormula("fSDDProtLandauWidth","[0]/(x*x)+[1]");
  fSDDProtLandauWidth->SetParameters(fSDDProtLWVParams);

  if(fSDDProtGaussWidth) delete fSDDProtGaussWidth;
  fSDDProtGaussWidth=new TFormula("fSDDProtGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDProtGaussWidth->SetParameters(fSDDProtGWVParams);

  if(fSSDProtMPV) delete fSSDProtMPV;
  fSSDProtMPV=new TFormula("fSSDProtMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDProtMPV->SetParameters(fSSDProtMPVParams);

  if(fSSDProtLandauWidth) delete fSSDProtLandauWidth;
  fSSDProtLandauWidth=new TFormula("fSSDProtLandauWidth","[0]/(x*x)+[1]");
  fSSDProtLandauWidth->SetParameters(fSSDProtLWVParams);

  if(fSSDProtGaussWidth) delete fSSDProtGaussWidth;
  fSSDProtGaussWidth=new TFormula("fSSDProtGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDProtGaussWidth->SetParameters(fSSDProtGWVParams);
}
//______________________________________________________________________
void AliITSPidParams::InitData(){
  // initialize TFormulas to Real Data values (=p-p ALICE Experiment)
  // parameter values from LHC10b
  // MPV BetheBloch parameters;

  //sdd data electrons parameters
  Double_t fSDDElecMPVParams[2] = {  0.8736, 86.73};
  Double_t fSDDElecLWVParams[2] = { -0.3681, 6.611};
  Double_t fSDDElecGWVParams[2] = { 0.9324, 10.14};

  //ssd data electrons parameters
  Double_t fSSDElecMPVParams[2] = {  -0.09946, 89};
  Double_t fSSDElecLWVParams[2] = {  -0.05756, 6.236};
  Double_t fSSDElecGWVParams[2] = {    0.3355, 7.985};

  // electrons
  if(fSDDElecMPV) delete fSDDElecMPV;
  fSDDElecMPV=new TFormula("fSDDElecMPV","[0]*x+[1]");
  fSDDElecMPV->SetParameters(fSDDElecMPVParams);

  if(fSDDElecLandauWidth) delete fSDDElecLandauWidth;
  fSDDElecLandauWidth=new TFormula("fSDDElecLandauWidth","[0]*x+[1]");
  fSDDElecLandauWidth->SetParameters(fSDDElecLWVParams);

  if(fSDDElecGaussWidth) delete fSDDElecGaussWidth;
  fSDDElecGaussWidth=new TFormula("fSDDElecGaussWidth","[0]*x+[1]");
  fSDDElecGaussWidth->SetParameters(fSDDElecGWVParams);

  if(fSSDElecMPV) delete fSSDElecMPV;
  fSSDElecMPV=new TFormula("fSSDElecMPV","[0]*x+[1]");
  fSSDElecMPV->SetParameters(fSSDElecMPVParams);

  if(fSSDElecLandauWidth) delete fSSDElecLandauWidth;
  fSSDElecLandauWidth=new TFormula("fSSDElecLandauWidth","[0]*x+[1]");
  fSSDElecLandauWidth->SetParameters(fSSDElecLWVParams);

  if(fSSDElecGaussWidth) delete fSSDElecGaussWidth;
  fSSDElecGaussWidth=new TFormula("fSSDElecGaussWidth","[0]*x+[1]");
  fSSDElecGaussWidth->SetParameters(fSSDElecGWVParams);

  //sdd data hadrons parameters
  //pion
  Double_t fSDDPionMPVParams[3] = { 1.348,  5.457, 80.29};
  Double_t fSDDPionLWVParams[3] = {0.1526, 0.2125, 6.456};
  Double_t fSDDPionGWVParams[3] = {0.2112,   1.07, 8.882};
  //kaon
  Double_t fSDDKaonMPVParams[3] = { 13.35, -8.146, 74.82};
  Double_t fSDDKaonLWVParams[2] = { 1.247,  5.966};
  Double_t fSDDKaonGWVParams[3] = { 4.649,   6.21, 5.832};
  //proton
  Double_t fSDDProtMPVParams[3] = {-2.753, -136.5, 133.5};
  Double_t fSDDProtLWVParams[2] = { 5.393, 4.793};
  Double_t fSDDProtGWVParams[3] = { 21.08,   30.0, -4.265};

  //ssd data hadrons parameters
  //pion
  Double_t fSSDPionMPVParams[3] = { 1.435,  5.768, 81.76};
  Double_t fSSDPionLWVParams[3] = {0.2191,  0.385, 6.207};
  Double_t fSSDPionGWVParams[3] = {0.1941, 0.9167, 6.712};
  //kaon
  Double_t fSSDKaonMPVParams[3] = { 18.71, 4.229, 71.27};
  Double_t fSSDKaonLWVParams[2] = { 1.421,  5.547};
  Double_t fSSDKaonGWVParams[3] = { 6.208, 10.35, 1.885};
  //proton
  Double_t fSSDProtMPVParams[3] = {9.856, -108.9, 122.5};
  Double_t fSSDProtLWVParams[2] = { 5.61,  4.315};
  Double_t fSSDProtGWVParams[3] = {37.12,  65.45, -23.83};

 // pions
  if(fSDDPionMPV) delete fSDDPionMPV;
  fSDDPionMPV=new TFormula("fSDDPionMPV","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSDDPionMPV->SetParameters(fSDDPionMPVParams);

  if(fSDDPionLandauWidth) delete fSDDPionLandauWidth;
  fSDDPionLandauWidth=new TFormula("fSDDPionLandauWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDPionLandauWidth->SetParameters(fSDDPionLWVParams);

  if(fSDDPionGaussWidth) delete fSDDPionGaussWidth;
  fSDDPionGaussWidth=new TFormula("fSDDPionGaussWidth","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSDDPionGaussWidth->SetParameters(fSDDPionGWVParams);

  if(fSSDPionMPV) delete fSSDPionMPV;
  fSSDPionMPV=new TFormula("fSSDPionMPV","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSSDPionMPV->SetParameters(fSSDPionMPVParams);

  if(fSSDPionLandauWidth) delete fSSDPionLandauWidth;
  fSSDPionLandauWidth=new TFormula("fSSDPionLandauWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDPionLandauWidth->SetParameters(fSSDPionLWVParams);

  if(fSSDPionGaussWidth) delete fSSDPionGaussWidth;
  fSSDPionGaussWidth=new TFormula("fSSDPionGaussWidth","[0]/(x*x)+[1]*TMath::Log(x)+[2]");
  fSSDPionGaussWidth->SetParameters(fSSDPionGWVParams);

  // kaons
  if(fSDDKaonMPV) delete fSDDKaonMPV;
  fSDDKaonMPV=new TFormula("fSDDKaonMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDKaonMPV->SetParameters(fSDDKaonMPVParams);

  if(fSDDKaonLandauWidth) delete fSDDKaonLandauWidth;
  fSDDKaonLandauWidth=new TFormula("fSDDKaonLandauWidth","[0]/(x*x)+[1]");
  fSDDKaonLandauWidth->SetParameters(fSDDKaonLWVParams);

  if(fSDDKaonGaussWidth) delete fSDDKaonGaussWidth;
  fSDDKaonGaussWidth=new TFormula("fSDDKaonGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDKaonGaussWidth->SetParameters(fSDDKaonGWVParams);

  if(fSSDKaonMPV) delete fSSDKaonMPV;
  fSSDKaonMPV=new TFormula("fSSDKaonMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDKaonMPV->SetParameters(fSSDKaonMPVParams);

  if(fSSDKaonLandauWidth) delete fSSDKaonLandauWidth;
  fSSDKaonLandauWidth=new TFormula("fSSDKaonLandauWidth","[0]/(x*x)+[1]");
  fSSDKaonLandauWidth->SetParameters(fSSDKaonLWVParams);

  if(fSSDKaonGaussWidth) delete fSSDKaonGaussWidth;
  fSSDKaonGaussWidth=new TFormula("fSSDKaonGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDKaonGaussWidth->SetParameters(fSSDKaonGWVParams);

  // protons
  if(fSDDProtMPV) delete fSDDProtMPV;
  fSDDProtMPV=new TFormula("fSDDProtMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDProtMPV->SetParameters(fSDDProtMPVParams);

  if(fSDDProtLandauWidth) delete fSDDProtLandauWidth;
  fSDDProtLandauWidth=new TFormula("fSDDProtLandauWidth","[0]/(x*x)+[1]");
  fSDDProtLandauWidth->SetParameters(fSDDProtLWVParams);

  if(fSDDProtGaussWidth) delete fSDDProtGaussWidth;
  fSDDProtGaussWidth=new TFormula("fSDDProtGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSDDProtGaussWidth->SetParameters(fSDDProtGWVParams);

  if(fSSDProtMPV) delete fSSDProtMPV;
  fSSDProtMPV=new TFormula("fSSDProtMPV","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDProtMPV->SetParameters(fSSDProtMPVParams);

  if(fSSDProtLandauWidth) delete fSSDProtLandauWidth;
  fSSDProtLandauWidth=new TFormula("fSSDProtLandauWidth","[0]/(x*x)+[1]");
  fSSDProtLandauWidth->SetParameters(fSSDProtLWVParams);

  if(fSSDProtGaussWidth) delete fSSDProtGaussWidth;
  fSSDProtGaussWidth=new TFormula("fSSDProtGaussWidth","[0]/(x*x)+[1]/x*TMath::Log(x)+[2]");
  fSSDProtGaussWidth->SetParameters(fSSDProtGWVParams);
}
//_______________________________________________________________________
Double_t AliITSPidParams::GetLandauGausNormPdgCode(Double_t dedx, Int_t pdgCode, Double_t mom, Int_t lay) const {
  // Computes Landau Gauss convolution for given particle specie and given momentum in a given ITS layer
  if(TMath::Abs(pdgCode)==11) return GetLandauGausNorm(dedx,AliPID::kElectron,mom,lay);
  else if(TMath::Abs(pdgCode)==211) return GetLandauGausNorm(dedx,AliPID::kPion,mom,lay);
  else if(TMath::Abs(pdgCode)==321) return GetLandauGausNorm(dedx,AliPID::kKaon,mom,lay);
  else if(TMath::Abs(pdgCode)==2212) return GetLandauGausNorm(dedx,AliPID::kProton,mom,lay);
  else return 0.;
}
//_______________________________________________________________________
Double_t AliITSPidParams::GetLandauGausNorm(Double_t dedx, Int_t partType, Double_t mom, Int_t lay) const{
  // Computes Landau Gauss convolution for given particle specie and given momentum in a given ITS layer

  Double_t par[4];
  Bool_t isSet=kFALSE;
  if(partType==AliPID::kElectron){
    if(lay==3 || lay==4){
      par[0]=GetSDDElecLandauWidth(mom);
      par[1]=GetSDDElecMPV(mom);
      par[2]=GetSDDElecGaussWidth(mom);
      isSet=kTRUE;
    }
    else if(lay==5 || lay==6){
      par[0]=GetSSDElecLandauWidth(mom);
      par[1]=GetSSDElecMPV(mom);
      par[2]=GetSSDElecGaussWidth(mom);
      isSet=kTRUE;
    }
  }else if(partType==AliPID::kPion){
    if(lay==3 || lay==4){
      par[0]=GetSDDPionLandauWidth(mom);
      par[1]=GetSDDPionMPV(mom);
      par[2]=GetSDDPionGaussWidth(mom);
      isSet=kTRUE;
    }
    else if(lay==5 || lay==6){
      par[0]=GetSSDPionLandauWidth(mom);
      par[1]=GetSSDPionMPV(mom);
      par[2]=GetSSDPionGaussWidth(mom);
      isSet=kTRUE;
    }
  }else if(partType==AliPID::kKaon){
    if(lay==3 || lay==4){
      par[0]=GetSDDKaonLandauWidth(mom);
      par[1]=GetSDDKaonMPV(mom);
      par[2]=GetSDDKaonGaussWidth(mom);
      isSet=kTRUE;
    }
    else if(lay==5 || lay==6){
      par[0]=GetSSDKaonLandauWidth(mom);
      par[1]=GetSSDKaonMPV(mom);
      par[2]=GetSSDKaonGaussWidth(mom);
      isSet=kTRUE;
    }
  }else if(partType==AliPID::kProton){
    if(lay==3 || lay==4){
      par[0]=GetSDDProtLandauWidth(mom);
      par[1]=GetSDDProtMPV(mom);
      par[2]=GetSDDProtGaussWidth(mom);
      isSet=kTRUE;
    }
    else if(lay==5 || lay==6){
      par[0]=GetSSDProtLandauWidth(mom);
      par[1]=GetSSDProtMPV(mom);
      par[2]=GetSSDProtGaussWidth(mom);
      isSet=kTRUE;
    }
  }
  if(!isSet) return 0.;
  // Numeric constants
  Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
  Double_t mpshift  = -0.22278298;       // Landau maximum location
  // Control constants
  Double_t np = 100.0;      // number of convolution steps
  Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas
  // Variables
  Double_t xx;
  Double_t mpc;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow,xupp;
  Double_t step;
  Double_t i;

  // MP shift correction
  mpc = par[1] - mpshift * par[0];
  // Range of convolution integral
  xlow = dedx - sc * par[2];
  xupp = dedx + sc * par[2];
  if(np!=0) step = (xupp-xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for(i=1.0; i<=np/2; i++) {
    xx = xlow + (i-.5) * step;

    fland = TMath::Landau(xx,mpc,par[0]) / par[0];
    sum += fland * TMath::Gaus(dedx,xx,par[2]);

    xx = xupp - (i-.5) * step;
    fland = TMath::Landau(xx,mpc,par[0]) / par[0];
    sum += fland * TMath::Gaus(dedx,xx,par[2]);
  }

  return (step * sum * invsq2pi / par[2]);
}

