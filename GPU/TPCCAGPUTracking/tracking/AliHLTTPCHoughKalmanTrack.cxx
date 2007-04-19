// @(#) $Id$
// origin: hough/AliL3HoughKalmanTrack.cxx,v 1.3 Tue Sep 5 08:45:27 2006 UTC by hristov

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

//-------------------------------------------------------------------------
//          Implementation of the HLT TPC Hough Kalman track class
//
//          Origin: Cvetan Cheshkov, CERN, Cvetan.Cheshkov@cern.ch
//-------------------------------------------------------------------------

#include "AliHLTTPCHoughKalmanTrack.h"

#include "AliHLTStdIncludes.h"
#include "AliHLTTPCHoughTrack.h"
#include "AliHLTTPCHoughTransformer.h"
#include "AliHLTTPCHoughTransformerRow.h"
#include "AliHLTTPCHistogram.h"

Int_t CalcExternalParams(const AliHLTTPCHoughTrack& t, Double_t deltax, Double_t deltay, Double_t deltaeta, const Double_t zvertex, const Double_t xhit, Double_t xx[5]);

ClassImp(AliHLTTPCHoughKalmanTrack)

//____________________________________________________________________________
AliHLTTPCHoughKalmanTrack::AliHLTTPCHoughKalmanTrack(const AliHLTTPCHoughTrack& t) throw (const Char_t *) 
              : AliTPCtrack() 
{
  // The method constructs an AliHLTTPCHoughKalmanTrack object
  // from an HLT TPC Hough track

  SetChi2(0.);
  SetNumberOfClusters(t.GetLastRow()-t.GetFirstRow());
  SetLabel(t.GetMCid());
  SetFakeRatio(0.);
  SetMass(0.13957);

  fdEdx=0;
  Double_t alpha = fmod((t.GetSector()+0.5)*(2*TMath::Pi()/18),2*TMath::Pi());
  if      (alpha < -TMath::Pi()) alpha += 2*TMath::Pi();
  else if (alpha >= TMath::Pi()) alpha -= 2*TMath::Pi();

  const Double_t xhit = 82.97;
  const Double_t zvertex = t.GetFirstPointZ();
  Double_t par[5];
  Double_t deltax = t.GetPterr();
  Double_t deltay = t.GetPsierr();
  Double_t deltaeta = t.GetTglerr();
  if(CalcExternalParams(t,0,0,0,zvertex,xhit,par)==0) throw "AliHLTTPCHoughKalmanTrack: conversion failed !\n";

  Double_t cnv=1./(GetBz()*kB2C);

  //and covariance matrix
  //For the moment estimate the covariance matrix numerically
  Double_t xx1[5];
  if(CalcExternalParams(t,deltax,0,0,zvertex,xhit,xx1)==0) throw "AliHLTTPCHoughKalmanTrack: conversion failed !\n";
  Double_t xx2[5];
  if(CalcExternalParams(t,0,deltay,0,zvertex,xhit,xx2)==0) throw "AliHLTTPCHoughKalmanTrack: conversion failed !\n";
  Double_t xx3[5];
  if(CalcExternalParams(t,0,0,deltaeta,zvertex,xhit,xx3)==0) throw "AliHLTTPCHoughKalmanTrack: conversion failed !\n";

  Double_t dx1[5],dx2[5],dx3[5];
  for(Int_t i=0;i<5;i++) {
    dx1[i]=xx1[i]-par[i];
    dx2[i]=xx2[i]-par[i];
    dx3[i]=xx3[i]-par[i];
  }

  Double_t cov[15]={
    dx1[0]*dx1[0]+dx2[0]*dx2[0],
    0.,  dx3[1]*dx3[1],
    0.,  0.,  dx1[2]*dx1[2]+dx2[2]*dx2[2],
    0.,  dx3[3]*dx3[1],  0.,  dx3[3]*dx3[3],
    0.,  0.,  0.,  0.,  dx1[4]*dx1[4]+dx2[4]*dx2[4]
  };
  /*
  fC20=dx1[2]*dx1[0]+dx2[2]*dx2[0];
  fC40=dx1[4]*dx1[0]+dx2[4]*dx2[0];
  fC42=dx1[4]*dx1[2]+dx2[4]*dx2[2];
  fC33=dx3[3]*dx3[3];
  fC11=dx3[1]*dx3[1];
  fC31=dx3[3]*dx3[1];
  fC10=fC30=fC21=fC41=fC32=fC43=0;
  fC20=fC42=fC40=0;
  */
  cov[10]*=cnv; cov[11]*=cnv; cov[12]*=cnv; cov[13]*=cnv; cov[14]*=(cnv*cnv); 
  par[4]*=cnv;

  Set(xhit,alpha,par,cov);

}

//____________________________________________________________________________
Int_t CalcExternalParams(const AliHLTTPCHoughTrack& t, Double_t deltax, Double_t deltay, Double_t deltaeta, const Double_t zvertex, const Double_t xhit, Double_t xx[5])
{
  // Translate the parameters of the Hough tracks into
  // AliKalmanTrack paramters

  //First get the emiision angle and track curvature
  Double_t binx = t.GetBinX()+deltax;
  Double_t biny = t.GetBinY()+deltay;
  Double_t psi = atan((binx-biny)/(AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2()));
  Double_t kappa = 2.0*(binx*cos(psi)-AliHLTTPCHoughTransformerRow::GetBeta1()*sin(psi));
  Double_t radius = 1./kappa;

  //Local y coordinate
  Double_t centerx = -1.*radius*sin(psi);
  Double_t centery = radius*cos(psi);
  Double_t aa = (xhit - centerx)*(xhit - centerx);
  Double_t r2 = radius*radius;
  if(aa > r2) return 0;
  Double_t aa2 = sqrt(r2 - aa);
  Double_t y1 = centery + aa2;
  Double_t y2 = centery - aa2;
  Double_t yhit = y1;
  if(fabs(y2) < fabs(y1)) yhit = y2;

  //Local z coordinate
  Double_t stot = sqrt(xhit*xhit+yhit*yhit);
  Double_t zhit;

  //Lambda
  Double_t eta=t.GetPseudoRapidity()+deltaeta;
  Double_t theta = 2*atan(exp(-1.*eta));
  Double_t tanl = 1./tan(theta);
  zhit = zvertex + stot*tanl;

  xx[0] = yhit;
  xx[1] = zhit;
  xx[2] = (xhit-centerx)/radius;
  xx[3] = tanl;
  xx[4] = kappa;
  return 1;
}
