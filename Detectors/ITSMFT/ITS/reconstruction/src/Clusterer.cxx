// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.cxx
/// \brief Implementation of the ITS cluster finder
#include <algorithm>
#include "FairLogger.h"      // for LOG

#include "TClonesArray.h"

#include "ITSReconstruction/Clusterer.h"
#include "ITSReconstruction/Cluster.h"

using namespace o2::ITS;
using namespace o2::ITSMFT;

Float_t Clusterer::mPitchX=0.002;
Float_t Clusterer::mPitchZ=0.002;
Float_t Clusterer::mX0=0.;
Float_t Clusterer::mZ0=0.;

Clusterer::Clusterer()
  :mCurr(mColumn2+1)
  ,mPrev(mColumn1+1)
  ,mChipID(65535)
  ,mCol(65535)
{
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);
}

void Clusterer::process(PixelReader &reader, TClonesArray &clusters)
{
  reader.init();

  while (reader.getNextChipData(mChipData)) {
    LOG(DEBUG) <<"ITSClusterer got Chip " << mChipData.chipID << " ROFrame " << mChipData.roFrame
	       << " Nhits " << mChipData.pixels.size() << FairLogger::endl;;
    initChip();
    for (int ip=1;ip<mChipData.pixels.size();ip++) updateChip(ip);
    finishChip(clusters);
  }

}

void Clusterer::initChip()
{
  mPrev=mColumn1+1;
  mCurr=mColumn2+1;
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);

  mPixels.clear();
  mPreClusterHeads.clear();
  mPreClusterIndices.clear();
  PixelReader::PixelData* pix = &mChipData.pixels[0]; 
  mCol = pix->col;
  mCurr[pix->row] = 0;
  //start the first pre-cluster
  mPreClusterHeads.push_back(0);
  mPreClusterIndices.push_back(0);
  mPixels.emplace_back(-1,pix);
}

void Clusterer::updateChip(int ip)
{
  PixelReader::PixelData* pix = &mChipData.pixels[ip]; 
  if (mCol != pix->col) { // switch the buffers
    Int_t *tmp = mCurr;
    mCurr = mPrev;
    mPrev = tmp;
    if (pix->col > mCol+1) std::fill(mPrev, mPrev+kMaxRow, -1);
    std::fill(mCurr, mCurr+kMaxRow, -1);
    mCol = pix->col;
  }

  Bool_t attached=false;
  UShort_t row = pix->row;
  Int_t neighbours[]{mCurr[row-1], mPrev[row], mPrev[row+1], mPrev[row-1]};
  for (auto pci : neighbours) {
     if (pci<0) continue;
     auto &ci = mPreClusterIndices[pci];
     if (attached) {
        auto &newci = mPreClusterIndices[mCurr[row]];
	if (ci < newci) newci = ci;
	else ci = newci;
     } else {
       auto &firstIndex = mPreClusterHeads[ci];
        mPixels.emplace_back(firstIndex, pix);
        firstIndex = mPixels.size() - 1;
	mCurr[row] = pci;
	attached = true;
     }
  }
  
  if (attached) return;

  //start new precluster
  mPreClusterHeads.push_back(mPixels.size());
  mPixels.emplace_back(-1,pix);
  Int_t lastIndex = mPreClusterIndices.size();
  mPreClusterIndices.push_back(lastIndex);
  mCurr[row] = lastIndex;

}

void Clusterer::finishChip(TClonesArray &clusters)
{
  static Float_t sigmaX2 = mPitchX * mPitchX / 12.; //FIXME
  static Float_t sigmaY2 = mPitchZ * mPitchZ / 12.;

  Int_t noc = clusters.GetEntriesFast();  
  for (Int_t i1=0; i1<mPreClusterHeads.size(); ++i1) {
    const auto ci = mPreClusterIndices[i1];
    if (ci<0) continue;
    UShort_t xmax=0, xmin=65535;
    UShort_t zmax=0, zmin=65535;
    Float_t x=0., z=0.;
    int labels[Cluster::maxLabels], nlab = 0, npix = 0;
    Int_t next = mPreClusterHeads[i1];
    while (next >= 0) {
      const auto &dig = mPixels[next];
      const auto pix = dig.second; // PixelReader.PixelData*
      x += pix->row;
      z += pix->col;
      if (pix->row < xmin) xmin = pix->row;
      if (pix->row > xmax) xmax = pix->row;
      if (pix->col < zmin) zmin = pix->col;
      if (pix->col > zmax) zmax = pix->col;
      //
      // add labels
      fetchMCLabels(pix, labels, nlab);      
      npix++;
      next = dig.first;
    }
    mPreClusterIndices[i1] = -1;
    for (Int_t i2=i1+1; i2<mPreClusterHeads.size(); ++i2) {
      if (mPreClusterIndices[i2] != ci) continue;
      next = mPreClusterHeads[i2];
      while (next >= 0) {
        const auto &dig = mPixels[next];
	const auto pix = dig.second; // PixelReader.PixelData*
	x += pix->row;
	z += pix->col;
	if (pix->row < xmin) xmin = pix->row;
	if (pix->row > xmax) xmax = pix->row;
	if (pix->col < zmin) zmin = pix->col;
	if (pix->col > zmax) zmax = pix->col;
	// add labels
	fetchMCLabels(pix, labels, nlab);   
        npix++;
	next = dig.first;
      }
      mPreClusterIndices[i2] = -1;
    }    
    x /= npix;
    x = mX0 + x*mPitchX;
    z /= npix;
    z = mZ0 + z*mPitchZ;
    Cluster *c = static_cast<Cluster *>(clusters.ConstructedAt(noc++));
    c->setVolumeId(mChipData.chipID);
    c->setROFrame(mChipData.roFrame);
    c->SetTimeStamp(mChipData.timeStamp);
    c->setX(x);
    c->setY(0);
    c->setZ(z);
    c->setSigmaY2(sigmaX2);
    c->setSigmaZ2(sigmaY2);
    c->setNxNzN(xmax-xmin+1,zmax-zmin+1,npix);
    c->setFrameLoc();
    for (int i=nlab;i--;) c->setLabel(labels[i],i);
  }
}

void Clusterer::fetchMCLabels(const PixelReader::PixelData* pix, int *labels, int &nfilled) const
{
  // transfer MC labels to cluster
  if (nfilled>=Cluster::maxLabels) return;
  int lbl;
  for (int id=0;id<Digit::maxLabels;id++) {
    if ((lbl=pix->labels[id])<0) return; // all following labels will be 0
    int ic = nfilled;
    for (;ic--;) { // check if the label is already present
      if (labels[ic]==lbl) break;
    }
    if (ic<0) { // label not found
      labels[nfilled++] = lbl;
      if (nfilled>=Cluster::maxLabels) break;
    }
  }
  //
}
