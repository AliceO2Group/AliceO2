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

#include "AliGRPRecoParam.h"
#include "AliV0HypSel.h"
#include "AliLog.h"

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Class with GRP reconstruction parameters                                  //
// Origin: andrea.dainese@lnl.infn.it                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////



ClassImp(AliGRPRecoParam)

//_____________________________________________________________________________
AliGRPRecoParam::AliGRPRecoParam() : AliDetectorRecoParam(),
fMostProbablePt(0.350),
fVertexerTracksConstraintITS(kTRUE),
fVertexerTracksConstraintTPC(kTRUE),
fVertexerTracksNCuts(24),
fVertexerTracksITSdcacut(0.1),
fVertexerTracksITSdcacutIter0(0.1),
fVertexerTracksITSmaxd0z0(0.5),
fVertexerTracksITSminCls(5),
fVertexerTracksITSmintrks(1),
fVertexerTracksITSnsigma(3.),
fVertexerTracksITSnindetfitter(100.),
fVertexerTracksITSmaxtgl(1000.), 
fVertexerTracksITSfidR(3.),
fVertexerTracksITSfidZ(30.),
fVertexerTracksITSalgo(1.),
fVertexerTracksITSalgoIter0(4.),
//
fVertexerTracksITSMVTukey2(7.),
fVertexerTracksITSMVSig2Ini(1e3),
fVertexerTracksITSMVMaxSigma2(5.0),
fVertexerTracksITSMVMinSig2Red(0.05),
fVertexerTracksITSMVMinDst(10e-4),
fVertexerTracksITSMVScanStep(2.),
fVertexerTracksITSMVMaxWghNtr(10),
fVertexerTracksITSMVFinalWBinary(1),
fVertexerTracksITSMVBCSpacing(50),
//
fVertexerTracksITSclusterize(0),
fVertexerTracksITSclusterdz(999999.),
fVertexerTracksITSclusternsigmaz(3.),
//
fVertexerTracksTPCdcacut(0.1),
fVertexerTracksTPCdcacutIter0(1.0),
fVertexerTracksTPCmaxd0z0(5.),
fVertexerTracksTPCminCls(10),
fVertexerTracksTPCmintrks(1),
fVertexerTracksTPCnsigma(3.),
fVertexerTracksTPCnindetfitter(0.1),
fVertexerTracksTPCmaxtgl(1.5), 
fVertexerTracksTPCfidR(3.),
fVertexerTracksTPCfidZ(30.),
fVertexerTracksTPCalgo(1.),
fVertexerTracksTPCalgoIter0(4.),
//
fVertexerTracksTPCMVTukey2(7.),
fVertexerTracksTPCMVSig2Ini(1e3),
fVertexerTracksTPCMVMaxSigma2(5.0),
fVertexerTracksTPCMVMinSig2Red(0.05),
fVertexerTracksTPCMVMinDst(10e-4),
fVertexerTracksTPCMVScanStep(2.),
fVertexerTracksTPCMVMaxWghNtr(10),
fVertexerTracksTPCMVFinalWBinary(1),
fVertexerTracksTPCMVBCSpacing(50),
//
fVertexerTracksTPCclusterize(0),
fVertexerTracksTPCclusterdz(999999.),
fVertexerTracksTPCclusternsigmaz(3.),
//
fVertexerV0NCuts(8),
fVertexerV0Chi2max(33.),
fVertexerV0DNmin(0.05),
fVertexerV0DPmin(0.05),
fVertexerV0DCAmax(1.5),
fVertexerV0CPAmin(0.9),
fVertexerV0Rmin(0.2),
fVertexerV0Rmax(200.),
fVertexerV0EtaMax(5.0),
fCleanOfflineV0Prongs(kFALSE),
fVertexerCascadeNCuts(8),
fVertexerCascadeChi2max(33.),
fVertexerCascadeDV0min(0.01),
fVertexerCascadeMassWin(0.008),
fVertexerCascadeDBachMin(0.01),
fVertexerCascadeDCAmax(2.0),
fVertexerCascadeCPAmin(0.98),
fVertexerCascadeRmin(0.2),
fVertexerCascadeRmax(100.),
fCleanDCAZCut(-1),
fFlagsNotToClean(0xffffffffffffffff),
fV0HypSelArray()

{
  //
  // constructor
  //
  SetName("GRP");
  SetTitle("GRP");
}

//_____________________________________________________________________________
AliGRPRecoParam::~AliGRPRecoParam() 
{
  //
  // destructor
  //  
}

AliGRPRecoParam::AliGRPRecoParam(const AliGRPRecoParam& par) :
  AliDetectorRecoParam(par),
  fMostProbablePt(par.fMostProbablePt),
  fVertexerTracksConstraintITS(par.fVertexerTracksConstraintITS),
  fVertexerTracksConstraintTPC(par.fVertexerTracksConstraintTPC),
  fVertexerTracksNCuts(par.fVertexerTracksNCuts),
  fVertexerTracksITSdcacut(par.fVertexerTracksITSdcacut),
  fVertexerTracksITSdcacutIter0(par.fVertexerTracksITSdcacutIter0),
  fVertexerTracksITSmaxd0z0(par.fVertexerTracksITSmaxd0z0),
  fVertexerTracksITSminCls(par.fVertexerTracksITSminCls),
  fVertexerTracksITSmintrks(par.fVertexerTracksITSmintrks),
  fVertexerTracksITSnsigma(par.fVertexerTracksITSnsigma),
  fVertexerTracksITSnindetfitter(par.fVertexerTracksITSnindetfitter),
  fVertexerTracksITSmaxtgl(par.fVertexerTracksITSmaxtgl), 
  fVertexerTracksITSfidR(par.fVertexerTracksITSfidR),
  fVertexerTracksITSfidZ(par.fVertexerTracksITSfidZ),
  fVertexerTracksITSalgo(par.fVertexerTracksITSalgo),
  fVertexerTracksITSalgoIter0(par.fVertexerTracksITSalgoIter0),
  //
  fVertexerTracksITSMVTukey2(par.fVertexerTracksITSMVTukey2),
  fVertexerTracksITSMVSig2Ini(par.fVertexerTracksITSMVSig2Ini),
  fVertexerTracksITSMVMaxSigma2(par.fVertexerTracksITSMVMaxSigma2),
  fVertexerTracksITSMVMinSig2Red(par.fVertexerTracksITSMVMinSig2Red),
  fVertexerTracksITSMVMinDst(par.fVertexerTracksITSMVMinDst),
  fVertexerTracksITSMVScanStep(par.fVertexerTracksITSMVScanStep),
  fVertexerTracksITSMVMaxWghNtr(par.fVertexerTracksITSMVMaxWghNtr),
  fVertexerTracksITSMVFinalWBinary(par.fVertexerTracksITSMVFinalWBinary),
  fVertexerTracksITSMVBCSpacing(par.fVertexerTracksITSMVBCSpacing),
  //
  fVertexerTracksITSclusterize(par.fVertexerTracksITSclusterize),
  fVertexerTracksITSclusterdz(par.fVertexerTracksITSclusterdz),
  fVertexerTracksITSclusternsigmaz(par.fVertexerTracksITSclusternsigmaz),
  //
  fVertexerTracksTPCdcacut(par.fVertexerTracksTPCdcacut),
  fVertexerTracksTPCdcacutIter0(par.fVertexerTracksTPCdcacutIter0),
  fVertexerTracksTPCmaxd0z0(par.fVertexerTracksTPCmaxd0z0),
  fVertexerTracksTPCminCls(par.fVertexerTracksTPCminCls),
  fVertexerTracksTPCmintrks(par.fVertexerTracksTPCmintrks),
  fVertexerTracksTPCnsigma(par.fVertexerTracksTPCnsigma),
  fVertexerTracksTPCnindetfitter(par.fVertexerTracksTPCnindetfitter),
  fVertexerTracksTPCmaxtgl(par.fVertexerTracksTPCmaxtgl), 
  fVertexerTracksTPCfidR(par.fVertexerTracksTPCfidR),
  fVertexerTracksTPCfidZ(par.fVertexerTracksTPCfidZ),
  fVertexerTracksTPCalgo(par.fVertexerTracksTPCalgo),
  fVertexerTracksTPCalgoIter0(par.fVertexerTracksTPCalgoIter0),
  //
  fVertexerTracksTPCMVTukey2(par.fVertexerTracksTPCMVTukey2),
  fVertexerTracksTPCMVSig2Ini(par.fVertexerTracksTPCMVSig2Ini),
  fVertexerTracksTPCMVMaxSigma2(par.fVertexerTracksTPCMVMaxSigma2),
  fVertexerTracksTPCMVMinSig2Red(par.fVertexerTracksTPCMVMinSig2Red),
  fVertexerTracksTPCMVMinDst(par.fVertexerTracksTPCMVMinDst),
  fVertexerTracksTPCMVScanStep(par.fVertexerTracksTPCMVScanStep),
  fVertexerTracksTPCMVMaxWghNtr(par.fVertexerTracksTPCMVMaxWghNtr),
  fVertexerTracksTPCMVFinalWBinary(par.fVertexerTracksTPCMVFinalWBinary),
  fVertexerTracksTPCMVBCSpacing(par.fVertexerTracksTPCMVBCSpacing),
  //
  fVertexerTracksTPCclusterize(par.fVertexerTracksTPCclusterize),
  fVertexerTracksTPCclusterdz(par.fVertexerTracksTPCclusterdz),
  fVertexerTracksTPCclusternsigmaz(par.fVertexerTracksTPCclusternsigmaz),
  //
  fVertexerV0NCuts(par.fVertexerV0NCuts),
  fVertexerV0Chi2max(par.fVertexerV0Chi2max),
  fVertexerV0DNmin(par.fVertexerV0DNmin),
  fVertexerV0DPmin(par.fVertexerV0DPmin),
  fVertexerV0DCAmax(par.fVertexerV0DCAmax),
  fVertexerV0CPAmin(par.fVertexerV0CPAmin),
  fVertexerV0Rmin(par.fVertexerV0Rmin),
  fVertexerV0Rmax(par.fVertexerV0Rmax),
  fVertexerV0EtaMax(par.fVertexerV0EtaMax),
  fCleanOfflineV0Prongs(par.fCleanOfflineV0Prongs),
  fVertexerCascadeNCuts(par.fVertexerCascadeNCuts),
  fVertexerCascadeChi2max(par.fVertexerCascadeChi2max),
  fVertexerCascadeDV0min(par.fVertexerCascadeDV0min),
  fVertexerCascadeMassWin(par.fVertexerCascadeMassWin),
  fVertexerCascadeDBachMin(par.fVertexerCascadeDBachMin),
  fVertexerCascadeDCAmax(par.fVertexerCascadeDCAmax),
  fVertexerCascadeCPAmin(par.fVertexerCascadeCPAmin),
  fVertexerCascadeRmin(par.fVertexerCascadeRmin),
  fVertexerCascadeRmax(par.fVertexerCascadeRmax),
  fCleanDCAZCut(par.fCleanDCAZCut),
  fFlagsNotToClean(par.fFlagsNotToClean)
{
  // copy constructor
  for (int i=0;i<par.fV0HypSelArray.GetEntriesFast();i++) {
    const AliV0HypSel* h = dynamic_cast<const AliV0HypSel*>( par.fV0HypSelArray.At(i) );
    if (!h) {
      AliFatal("Object provided as V0 hypothesis selection cut is not recognized");      
    }
    AddV0HypSel( *h );
  }
  fV0HypSelArray.SetOwner(kTRUE);
}

//_____________________________________________________________________________
AliGRPRecoParam& AliGRPRecoParam::operator = (const AliGRPRecoParam& par)
{
  // assignment operator

  if(&par == this) return *this;

  this->~AliGRPRecoParam();
  new(this) AliGRPRecoParam(par);
  return *this;
}

//_____________________________________________________________________________
AliGRPRecoParam *AliGRPRecoParam::GetHighFluxParam() 
{
  //
  // make default reconstruction  parameters for high flux env.
  //
  AliGRPRecoParam *param = new AliGRPRecoParam();

  // to speed up the vertexing in PbPb
  param->fVertexerTracksITSalgoIter0 = 1.;
  param->fVertexerTracksTPCalgoIter0 = 1.;

  // tighter selections for V0s
  param->fVertexerV0Chi2max = 33.;
  param->fVertexerV0DNmin   = 0.1;
  param->fVertexerV0DPmin   = 0.1;
  param->fVertexerV0DCAmax  = 1.0;
  param->fVertexerV0CPAmin  = 0.998;
  param->fVertexerV0Rmin    = 0.9;
  param->fVertexerV0Rmax    = 100.;

  // tighter selections for Cascades
  param->fVertexerCascadeChi2max  = 33.; 
  param->fVertexerCascadeDV0min   = 0.05;  
  param->fVertexerCascadeMassWin  = 0.008; 
  param->fVertexerCascadeDBachMin = 0.030;
  param->fVertexerCascadeDCAmax   = 0.3;  
  param->fVertexerCascadeCPAmin   = 0.999;  
  param->fVertexerCascadeRmin     = 0.9;    
  param->fVertexerCascadeRmax     = 100.;    

  return param;
}
//_____________________________________________________________________________
AliGRPRecoParam *AliGRPRecoParam::GetLowFluxParam() 
{
  //
  // make default reconstruction  parameters for low  flux env.
  //
  AliGRPRecoParam *param = new AliGRPRecoParam();
  return param;
}
//_____________________________________________________________________________
AliGRPRecoParam *AliGRPRecoParam::GetCosmicTestParam() 
{
  //
  // make default reconstruction  parameters for cosmics env.
  //
  AliGRPRecoParam *param = new AliGRPRecoParam();

  param->SetVertexerTracksConstraintITS(kFALSE);
  param->SetVertexerTracksConstraintTPC(kFALSE);
  param->SetMostProbablePt(3.0);

  return param;
}
//_____________________________________________________________________________
void AliGRPRecoParam::GetVertexerTracksCuts(Int_t mode,Double_t *cuts, int n) const {
  //
  // get cuts for ITS (0) or TPC (1) mode
  //
  if(mode==1) {
    if (n>0)  cuts[0] = fVertexerTracksTPCdcacut;
    if (n>1)  cuts[1] = fVertexerTracksTPCdcacutIter0;
    if (n>2)  cuts[2] = fVertexerTracksTPCmaxd0z0;
    if (n>3)  cuts[3] = fVertexerTracksTPCminCls;
    if (n>4)  cuts[4] = fVertexerTracksTPCmintrks;
    if (n>5)  cuts[5] = fVertexerTracksTPCnsigma;
    if (n>6)  cuts[6] = fVertexerTracksTPCnindetfitter;
    if (n>7)  cuts[7] = fVertexerTracksTPCmaxtgl; 
    if (n>8)  cuts[8] = fVertexerTracksTPCfidR;
    if (n>9)  cuts[9] = fVertexerTracksTPCfidZ;
    if (n>10) cuts[10]= fVertexerTracksTPCalgo;
    if (n>11) cuts[11]= fVertexerTracksTPCalgoIter0;
    //
    if (n>12)  cuts[12]= fVertexerTracksTPCMVTukey2;
    if (n>13)  cuts[13]= fVertexerTracksTPCMVSig2Ini;
    if (n>14)  cuts[14]= fVertexerTracksTPCMVMaxSigma2;
    if (n>15)  cuts[15]= fVertexerTracksTPCMVMinSig2Red;
    if (n>16)  cuts[16]= fVertexerTracksTPCMVMinDst;
    if (n>17)  cuts[17]= fVertexerTracksTPCMVScanStep;
    if (n>18)  cuts[18]= fVertexerTracksTPCMVMaxWghNtr;
    if (n>19)  cuts[19]= fVertexerTracksTPCMVFinalWBinary;
    if (n>20)  cuts[20]= fVertexerTracksTPCMVBCSpacing;
    //
    if (n>21)  cuts[21]= fVertexerTracksTPCclusterize;
    if (n>22)  cuts[22]= fVertexerTracksTPCclusterdz;
    if (n>23)  cuts[23]= fVertexerTracksTPCclusternsigmaz;
  } else {
    if (n>0 ) cuts[0] = fVertexerTracksITSdcacut;
    if (n>1 ) cuts[1] = fVertexerTracksITSdcacutIter0;
    if (n>2 ) cuts[2] = fVertexerTracksITSmaxd0z0;
    if (n>3 ) cuts[3] = fVertexerTracksITSminCls;
    if (n>4 ) cuts[4] = fVertexerTracksITSmintrks;
    if (n>5 ) cuts[5] = fVertexerTracksITSnsigma;
    if (n>6 ) cuts[6] = fVertexerTracksITSnindetfitter;
    if (n>7 ) cuts[7] = fVertexerTracksITSmaxtgl; 
    if (n>8 ) cuts[8] = fVertexerTracksITSfidR;
    if (n>9 ) cuts[9] = fVertexerTracksITSfidZ;
    if (n>10) cuts[10]= fVertexerTracksITSalgo;
    if (n>11) cuts[11]= fVertexerTracksITSalgoIter0;
    //
    if (n>12) cuts[12]= fVertexerTracksITSMVTukey2;
    if (n>13) cuts[13]= fVertexerTracksITSMVSig2Ini;
    if (n>14) cuts[14]= fVertexerTracksITSMVMaxSigma2;
    if (n>15) cuts[15]= fVertexerTracksITSMVMinSig2Red;
    if (n>16) cuts[16]= fVertexerTracksITSMVMinDst;
    if (n>17) cuts[17]= fVertexerTracksITSMVScanStep;
    if (n>18) cuts[18]= fVertexerTracksITSMVMaxWghNtr;
    if (n>19) cuts[19]= fVertexerTracksITSMVFinalWBinary;
    if (n>20) cuts[20]= fVertexerTracksITSMVBCSpacing;
    //
    if (n>21)  cuts[21]= fVertexerTracksITSclusterize;
    if (n>22)  cuts[22]= fVertexerTracksITSclusterdz;
    if (n>23)  cuts[23]= fVertexerTracksITSclusternsigmaz;
  }

  return;
}
//_____________________________________________________________________________
void AliGRPRecoParam::SetVertexerTracksCuts(Int_t mode,Int_t ncuts,Double_t* cuts) {
  //
  // set cuts for ITS (0) or TPC (1) mode
  //
  if(ncuts!=fVertexerTracksNCuts) {
    printf("AliGRPRecoParam: Number of AliVertexerTracks cuts is %d\n",fVertexerTracksNCuts);
    return;
  }

  if(mode==1) {
    if (ncuts>0) fVertexerTracksTPCdcacut = cuts[0];
    if (ncuts>1) fVertexerTracksTPCdcacutIter0 = cuts[1];
    if (ncuts>2) fVertexerTracksTPCmaxd0z0 = cuts[2];
    if (ncuts>3) fVertexerTracksTPCminCls = cuts[3];
    if (ncuts>4) fVertexerTracksTPCmintrks = cuts[4];
    if (ncuts>5) fVertexerTracksTPCnsigma = cuts[5];
    if (ncuts>6) fVertexerTracksTPCnindetfitter = cuts[6];
    if (ncuts>7) fVertexerTracksTPCmaxtgl = cuts[7]; 
    if (ncuts>8) fVertexerTracksTPCfidR = cuts[8];
    if (ncuts>9) fVertexerTracksTPCfidZ = cuts[9];
    if (ncuts>10) fVertexerTracksTPCalgo = cuts[10];
    if (ncuts>11) fVertexerTracksTPCalgoIter0 = cuts[11];
    //
    if (ncuts>12) fVertexerTracksTPCMVTukey2       = cuts[12];
    if (ncuts>13) fVertexerTracksTPCMVSig2Ini      = cuts[13];
    if (ncuts>14) fVertexerTracksTPCMVMaxSigma2    = cuts[14];
    if (ncuts>15) fVertexerTracksTPCMVMinSig2Red   = cuts[15];
    if (ncuts>16) fVertexerTracksTPCMVMinDst       = cuts[16];
    if (ncuts>17) fVertexerTracksTPCMVScanStep     = cuts[17];
    if (ncuts>18) fVertexerTracksTPCMVMaxWghNtr    = cuts[18];
    if (ncuts>19) fVertexerTracksTPCMVFinalWBinary = cuts[19];
    if (ncuts>20) fVertexerTracksTPCMVBCSpacing    = cuts[20];
    //
    if (ncuts>21) fVertexerTracksTPCclusterize     = cuts[21];
    if (ncuts>22) fVertexerTracksTPCclusterdz      = cuts[22];
    if (ncuts>23) fVertexerTracksTPCclusternsigmaz = cuts[23];
  } else {
    if (ncuts>0) fVertexerTracksITSdcacut = cuts[0];
    if (ncuts>1) fVertexerTracksITSdcacutIter0 = cuts[1];
    if (ncuts>2) fVertexerTracksITSmaxd0z0 = cuts[2];
    if (ncuts>3) fVertexerTracksITSminCls = cuts[3];
    if (ncuts>4) fVertexerTracksITSmintrks = cuts[4];
    if (ncuts>5) fVertexerTracksITSnsigma = cuts[5];
    if (ncuts>6) fVertexerTracksITSnindetfitter = cuts[6];
    if (ncuts>7) fVertexerTracksITSmaxtgl = cuts[7]; 
    if (ncuts>8) fVertexerTracksITSfidR = cuts[8];
    if (ncuts>9) fVertexerTracksITSfidZ = cuts[9];
    if (ncuts>10) fVertexerTracksITSalgo = cuts[10];
    if (ncuts>11) fVertexerTracksITSalgoIter0 = cuts[11];
    //
    if (ncuts>12) fVertexerTracksITSMVTukey2       = cuts[12];
    if (ncuts>13) fVertexerTracksITSMVSig2Ini      = cuts[13];
    if (ncuts>14) fVertexerTracksITSMVMaxSigma2    = cuts[14];
    if (ncuts>15) fVertexerTracksITSMVMinSig2Red   = cuts[15];
    if (ncuts>16) fVertexerTracksITSMVMinDst       = cuts[16];
    if (ncuts>17) fVertexerTracksITSMVScanStep     = cuts[17];
    if (ncuts>18) fVertexerTracksITSMVMaxWghNtr    = cuts[18];
    if (ncuts>19) fVertexerTracksITSMVFinalWBinary = cuts[19];
    if (ncuts>20) fVertexerTracksITSMVBCSpacing    = cuts[20];
    //
    if (ncuts>21) fVertexerTracksITSclusterize     = cuts[21];
    if (ncuts>22) fVertexerTracksITSclusterdz      = cuts[22];
    if (ncuts>23) fVertexerTracksITSclusternsigmaz = cuts[23];
  }
  //
  return;
}
//_____________________________________________________________________________
void AliGRPRecoParam::GetVertexerV0Cuts(Double_t *cuts) const {
  //
  // get cuts for AliV0vertexer
  //
  cuts[0] = fVertexerV0Chi2max;
  cuts[1] = fVertexerV0DNmin;
  cuts[2] = fVertexerV0DPmin;
  cuts[3] = fVertexerV0DCAmax;
  cuts[4] = fVertexerV0CPAmin;
  cuts[5] = fVertexerV0Rmin;
  cuts[6] = fVertexerV0Rmax;
  cuts[7] = fVertexerV0EtaMax;
  return;
}
//_____________________________________________________________________________
void AliGRPRecoParam::SetVertexerV0Cuts(Int_t ncuts,Double_t cuts[8]) {
  //
  // set cuts for AliV0vertexer
  //
  if(ncuts!=fVertexerV0NCuts) {
    printf("AliGRPRecoParam: Number of AliV0vertexer cuts is %d\n",fVertexerV0NCuts);
    return;
  }
  fVertexerV0Chi2max = cuts[0];
  fVertexerV0DNmin   = cuts[1];
  fVertexerV0DPmin   = cuts[2];
  fVertexerV0DCAmax  = cuts[3];
  fVertexerV0CPAmin  = cuts[4];
  fVertexerV0Rmin    = cuts[5];
  fVertexerV0Rmax    = cuts[6];
  fVertexerV0EtaMax  = cuts[7];
  return;
}
//_____________________________________________________________________________
void AliGRPRecoParam::GetVertexerCascadeCuts(Double_t *cuts) const {
  //
  // get cuts for AliCascadevertexer
  //
  cuts[0] = fVertexerCascadeChi2max;
  cuts[1] = fVertexerCascadeDV0min;
  cuts[2] = fVertexerCascadeMassWin;
  cuts[3] = fVertexerCascadeDBachMin;
  cuts[4] = fVertexerCascadeDCAmax;
  cuts[5] = fVertexerCascadeCPAmin;
  cuts[6] = fVertexerCascadeRmin;
  cuts[7] = fVertexerCascadeRmax;
  return;
}
//_____________________________________________________________________________
void AliGRPRecoParam::SetVertexerCascadeCuts(Int_t ncuts,Double_t cuts[8]) {
  //
  // set cuts for AliCascadeVertexer
  //
  if(ncuts!=fVertexerCascadeNCuts) {
    printf("AliGRPRecoParam: Number of AliCascadeVertexer cuts is %d\n",fVertexerCascadeNCuts);
    return;
  }
  fVertexerCascadeChi2max  = cuts[0];
  fVertexerCascadeDV0min   = cuts[1];
  fVertexerCascadeMassWin  = cuts[2];
  fVertexerCascadeDBachMin = cuts[3];
  fVertexerCascadeDCAmax   = cuts[4];
  fVertexerCascadeCPAmin   = cuts[5];
  fVertexerCascadeRmin     = cuts[6];
  fVertexerCascadeRmax     = cuts[7];
  return;
}

//_____________________________________________________________________________
void AliGRPRecoParam::AddV0HypSel(const AliV0HypSel& sel)
{
  fV0HypSelArray.AddLast(new AliV0HypSel(sel));
}
