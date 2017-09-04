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

#include "TClonesArray.h"

#include "ITSReconstruction/PixelReader.h"
#include "ITSReconstruction/Clusterer.h"
#include "ITSReconstruction/Cluster.h"

using namespace o2::ITS;

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

  UShort_t chipID,row,col;
  Int_t label;
  if ( !reader.getNextFiredPixel(chipID,row,col,label) ) return;
  initChip(chipID,row,col,label);   // mChipID=chipID and mCol=col

  while (reader.getNextFiredPixel(chipID,row,col,label)) {

    while (chipID == mChipID) {
        updateChip(chipID,row,col,label);
        if ( !reader.getNextFiredPixel(chipID,row,col,label) ) goto exit;
    }

    finishChip(clusters);
    initChip(chipID,row,col,label); // mChipID=chipID and mCol=col
  }

 exit:
  finishChip(clusters);
}

void Clusterer::initChip(UShort_t chipID, UShort_t row, UShort_t col, Int_t label)
{
  mPrev=mColumn1+1;
  mCurr=mColumn2+1;
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);

  mPixels.clear();
  mPreClusterHeads.clear();
  mPreClusterIndices.clear();

  mChipID=chipID;
  mCol=col;
  mCurr[row]=0;
  //start the first pre-cluster
  mPreClusterHeads.emplace_back(0,label);
  mPreClusterIndices.push_back(0);
  mPixels.emplace_back(-1,Pixel(row,col));
}

void Clusterer::updateChip(UShort_t chipID, UShort_t row, UShort_t col, Int_t label)
{
  if (mCol != col) { // switch the buffers
     Int_t *tmp=mCurr;
     mCurr=mPrev;
     mPrev=tmp;
     if (col > mCol+1) std::fill(mPrev, mPrev+kMaxRow, -1);
     std::fill(mCurr, mCurr+kMaxRow, -1);
     mCol=col;
  }

  Bool_t attached=false;
  Int_t neighbours[]{mCurr[row-1], mPrev[row], mPrev[row+1], mPrev[row-1]};
  for (auto pci : neighbours) {
     if (pci<0) continue;
     auto &ci = mPreClusterIndices[pci];
     if (attached) {
        auto &newci = mPreClusterIndices[mCurr[row]];
	if (ci < newci) newci = ci;
	else ci = newci;
     } else {
        auto &firstIndex = mPreClusterHeads[ci].first;
        mPixels.emplace_back(firstIndex, Pixel(row,col));
        firstIndex = mPixels.size() - 1;
	mCurr[row] = pci;
	attached = true;
     }
  }
  
  if (attached) return;

  //start new precluster
  mPreClusterHeads.emplace_back(mPixels.size(), label);
  mPixels.emplace_back(-1,Pixel(row,col));
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
    Int_t npix = 0;
    Int_t next = mPreClusterHeads[i1].first;
    while (next >= 0) {
      const auto &dig = mPixels[next];
      x += dig.second.first;
      z += dig.second.second;
      if (dig.second.first  < xmin) xmin=dig.second.first;
      if (dig.second.first  > xmax) xmax=dig.second.first;
      if (dig.second.second < zmin) zmin=dig.second.second;
      if (dig.second.second > zmax) zmax=dig.second.second;
      npix++;
      next = dig.first;
    }
    mPreClusterIndices[i1] = -1;
    for (Int_t i2=i1+1; i2<mPreClusterHeads.size(); ++i2) {
      if (mPreClusterIndices[i2] != ci) continue;
      next = mPreClusterHeads[i2].first;
      while (next >= 0) {
        const auto &dig = mPixels[next];
        x += dig.second.first;
        z += dig.second.second;
        if (dig.second.first  < xmin) xmin=dig.second.first;
        if (dig.second.first  > xmax) xmax=dig.second.first;
        if (dig.second.second < zmin) zmin=dig.second.second;
        if (dig.second.second > zmax) zmax=dig.second.second;
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
    c->setVolumeId(mChipID);
    c->setX(x);
    c->setY(0);
    c->setZ(z);
    c->setSigmaY2(sigmaX2);
    c->setSigmaZ2(sigmaY2);
    c->setNxNzN(xmax-xmin+1,zmax-zmin+1,npix);
    c->setFrameLoc();
    c->setLabel(mPreClusterHeads[i1].second, 0);
  }
}
