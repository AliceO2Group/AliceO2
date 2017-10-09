// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TRandom.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <TSeqCollection.h>
#include <climits>
#include <algorithm>

#include "FairLogger.h"

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/SimuClusterShaper.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSMFTSimulation/DigiParams.h"



using namespace o2::ITSMFT;

constexpr float sec2ns = 1e9;


//______________________________________________________________________
Double_t SimulationAlpide::computeIncidenceAngle(TLorentzVector dir) const
{
  Vector3D<float> glob(dir.Px()/dir.P(), dir.Py()/dir.P(), dir.Pz()/dir.P());
  auto loc = (*mMat)^(glob);
  Vector3D<float> normal(0.f,-1.f,0.f);
  return TMath::Abs(loc.Dot(normal)/TMath::Sqrt(loc.Mag2()));
}


//______________________________________________________________________
void SimulationAlpide::Hits2Digits(const SegmentationPixel *seg, Double_t eventTime, UInt_t &minFr, UInt_t &maxFr)
{
  Int_t nhits = GetNumberOfHits();

  // convert hits to digits, returning the min and max RO frames processed
  //
  // addNoise(mParam[Noise], seg);
  // RS: attention: the noise will be added just before transferring the digits of the given frame to the output,
  // because at this stage we don't know to which RO frame the noise should be added

  for (Int_t h = 0; h < nhits; ++h) {

    const Hit *hit = GetHitAt(h);
    double hTime0 = hit->GetTime()*sec2ns + eventTime - mParams->getTimeOffset(); // time from the RO start, in ns
    if (hTime0 > UINT_MAX) {
      LOG(WARNING) << "Hit RO Frame undefined: time: " << hTime0 << " is in far future: hitTime: "
		   << hit->GetTime() << " EventTime: " << eventTime << " ChipOffset: "
		   << mParams->getTimeOffset() << FairLogger::endl;
      return;
    }

    // calculate RO Frame for this hit
    if (hTime0<0) hTime0 = 0.;
    UInt_t roframe = static_cast<UInt_t>(hTime0/mParams->getROFrameLenght());
    if (roframe<minFr) minFr = roframe;
    if (roframe>maxFr) maxFr = roframe;

    // check if the hit time is not in the dead time of the chip
    // RS: info from L.Musa: no inefficiency on the frame boundary
    //if ( mParams->getROFrameLenght() - roframe%mParams->getROFrameLenght() < mParams->getROFrameDeadTime()) continue;

    switch (mParams->getHit2DigitsMethod()) {
    case DigiParams::p2dCShape :
      Hit2DigitsCShape(hit, roframe, eventTime, seg);
      break;
    case DigiParams::p2dSimple :
      Hit2DigitsSimple(hit, roframe, eventTime, seg);
      break;
    default:
      LOG(ERROR) << "Unknown point to digit mode " <<  mParams->getHit2DigitsMethod() << FairLogger::endl;
      break;
    }
  }
}

//________________________________________________________________________
void SimulationAlpide::Hit2DigitsCShape(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg)
{
  // convert single hit to digits with CShape generation method

  float nSteps = static_cast<Float_t>(mParams->getNSimSteps());
  Vector3D<float> xyzLocS( (*mMat)^(hit->GetPosStart()) ); // start position
  Vector3D<float> xyzLocE( (*mMat)^(hit->GetPos()) ); // end position
  Vector3D<float> step(xyzLocE);
  step -= xyzLocS;  
  step /= nSteps; // position increment at each step
  // the electrons will injected in the middle of each step
  xyzLocS += step/2;
  xyzLocE -= step/2;
  
  int rowS=-1,colS=-1,rowE=-1,colE=-1, nSkip=0;
  // get entrance pixel row and col
  while (!seg->localToDetector(xyzLocS.X(), xyzLocS.Z(), rowS, colS)) { // guard-ring ?
    if (++nSkip>=nSteps) return; // did not enter to sensitive matrix
    xyzLocS += step;
  }
  // get exit pixel row and col
  while (!seg->localToDetector(xyzLocE.X(), xyzLocE.Z(), rowE, colE)) { // guard-ring ?
    if (++nSkip>=nSteps) return; // did not enter to sensitive matrix
    xyzLocE += step;
  }
  // estimate the limiting min/max row and col where the non-0 response is possible
  if (rowS>rowE) std::swap(rowS,rowE);
  if (colS>colE) std::swap(colS,colE);
  rowS -= AlpideRespSimMat::NPix/2;
  rowE += AlpideRespSimMat::NPix/2;
  if (rowS<0) rowS = 0;
  if (rowE>seg->getNumberOfRows()) rowE = seg->getNumberOfRows()-1;
  colS -= AlpideRespSimMat::NPix/2;
  colE += AlpideRespSimMat::NPix/2;
  if (colS<0) colS = 0;
  if (colE>seg->getNumberOfColumns()) colE = seg->getNumberOfColumns()-1;
  int rowSpan = rowE-rowS+1, colSpan = colE-colS+1; // size of plaquet response is expected

  float respMatrix[rowSpan][colSpan]; // response accumulated here
  std::fill(&respMatrix[0][0],&respMatrix[0][0]+rowSpan*colSpan,0.f);
  
  float nElectrons = hit->GetEnergyLoss()*mParams->getEnergyToNElectrons();      // total number of deposited electrons
  nElectrons /= nSteps; // N electrons injected per step
  if (nSkip) nSteps -= nSkip;
  
  int rowPrev=-1, colPrev=-1, row, col;
  float cRowPix=0.f, cColPix=0.f; // local coordinated of the current pixel center

  const o2::ITSMFT::AlpideSimResponse* resp = mParams->getAlpSimResponse();
  
  for (int iStep=nSteps;iStep--;) {
    // Get the pixel ID
    seg->localToDetector(xyzLocS.X(), xyzLocS.Z(), row, col);
    if (row!=rowPrev || col!=colPrev) { // update pixel and coordinates of its center
      if (!seg->detectorToLocal(row, col, cRowPix, cColPix)) continue; // should not happen
      rowPrev = row;
      colPrev = col;
    }
    bool flipCol, flipRow;
    // note that response needs coordinates along column row (locX) (locZ) then depth (locY)
    auto rspmat = resp->getResponse(xyzLocS.X()-cRowPix, xyzLocS.Z()-cColPix, xyzLocS.Y(), flipRow, flipCol);
    //printf("#%d Lx:%e Lz:%e Ly:%e r:%d c:%d | dZ:%e dX:%e | %p\n",iStep,xyzLocS.X(), xyzLocS.Z(),xyzLocS.Y(),row,col,
    //	   xyzLocS.Z()-cColPix, xyzLocS.X()-cRowPix, rspmat);

    xyzLocS += step;
    if (!rspmat) continue;
    
    for (int irow=AlpideRespSimMat::NPix; irow--;) {
      int rowDest = row+irow-AlpideRespSimMat::NPix/2 - rowS; // destination row in the respMatrix
      if (rowDest<0 || rowDest>=rowSpan) continue;
      for (int icol=AlpideRespSimMat::NPix; icol--;) {
	int colDest = col+icol-AlpideRespSimMat::NPix/2 - colS; // destination column in the respMatrix
	if (colDest<0 || colDest>=colSpan) continue;
	
	respMatrix[rowDest][colDest] += rspmat->getValue(irow,icol,flipRow,flipCol);
      }
    }
  }

  double hTime  = hit->GetTime()*sec2ns + eventTime; // time in ns

  // fire the pixels assuming Poisson(n_response_electrons)
  for (int irow=rowSpan;irow--;) {
    for (int icol=colSpan;icol--;) {
      float nEleResp = respMatrix[irow][icol];
      if (!nEleResp) continue;
      int nEle = gRandom->Poisson(nElectrons*nEleResp);
      if (nEle)	addDigit(roFrame, irow+rowS, icol+colS, nEle, hit->getCombLabel(), hTime);
    }
  }
      
}

//________________________________________________________________________
void SimulationAlpide::Hit2DigitsSimple(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg)
{
  // convert single hit to digits with 1 to 1 mapping
  Point3D<float> glo( 0.5*(hit->GetX() + hit->GetStartX()),
		      0.5*(hit->GetY() + hit->GetStartY()),
		      0.5*(hit->GetZ() + hit->GetStartZ()) );
  auto loc = (*mMat)^( glo );

  int ix, iz;
  if (!seg->localToDetector(loc.X(), loc.Z(), ix, iz)) {
    LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
    return;
  }
  addDigit(roFrame, ix, iz, hit->GetEnergyLoss()*mParams->getEnergyToNElectrons(), hit->getCombLabel(), hit->GetTime()*sec2ns + eventTime);
}


//______________________________________________________________________
void SimulationAlpide::addNoise(const SegmentationPixel* seg, UInt_t rofMin, UInt_t rofMax)
{
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  constexpr float ns2sec = 1e-9;

  float mean = mParams->getNoisePerPixel()*seg->getNumberOfPads();
  float nel = mParams->getChargeThreshold()*1.1;  // RS: TODO: need realistic spectrum of noise abovee threshold

  for (UInt_t rof = rofMin; rof<=rofMax; rof++) {
    nhits = gRandom->Poisson(mean);
    double tstamp = mParams->getTimeOffset()+rof*mParams->getROFrameLenght(); // time in ns
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(seg->getNumberOfRows());
      col = gRandom->Integer(seg->getNumberOfColumns());
      // RS TODO: why the noise was added with 0 charge? It should be above the threshold!
      addDigit(rof, row, col, nel, Label(-1,0,0), tstamp);
    }
  }
}
