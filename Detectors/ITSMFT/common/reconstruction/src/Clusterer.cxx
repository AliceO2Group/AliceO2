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

#include "ITSMFTReconstruction/Clusterer.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "ITSMFTBase/SegmentationAlpide.h"

using namespace o2::ITSMFT;
using Segmentation = o2::ITSMFT::SegmentationAlpide;

//__________________________________________________
Clusterer::Clusterer()
  :mCurr(mColumn2+1)
  ,mPrev(mColumn1+1)
{
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);
}

//__________________________________________________
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

//__________________________________________________
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

//__________________________________________________
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

//__________________________________________________
void Clusterer::finishChip(TClonesArray &clusters)
{
  constexpr Float_t SigmaX2 = Segmentation::PitchRow*Segmentation::PitchRow / 12.; //FIXME
  constexpr Float_t SigmaY2 = Segmentation::PitchCol*Segmentation::PitchCol / 12.; //FIXME

  std::array<Label,Cluster::maxLabels> labels;
  
  Int_t noc = clusters.GetEntriesFast();  
  for (Int_t i1=0; i1<mPreClusterHeads.size(); ++i1) {
    const auto ci = mPreClusterIndices[i1];
    if (ci<0) continue;
    UShort_t xmax=0, xmin=65535;
    UShort_t zmax=0, zmin=65535;
    Float_t x=0., z=0.;
    int nlab = 0, npix = 0;
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

    Point3D<float> xyzLoc( Segmentation::getFirstRowCoordinate() + x*Segmentation::PitchRow/npix, 0.f,
			   Segmentation::getFirstColCoordinate() + z*Segmentation::PitchCol/npix );
    auto xyzTra = mGeometry->getMatrixT2L(mChipData.chipID)^(xyzLoc); // inverse transform from Local to Tracking frame
    Cluster *c = static_cast<Cluster *>(clusters.ConstructedAt(noc++));
    c->setROFrame(mChipData.roFrame);
    c->setSensorID(mChipData.chipID);
    c->setPos(xyzTra);
    c->setErrors(SigmaX2, SigmaY2, 0.f);
    c->setNxNzN(xmax-xmin+1,zmax-zmin+1,npix);
    for (int i=nlab;i--;) c->setLabel(labels[i],i);
  }
}

//__________________________________________________
void Clusterer::fetchMCLabels(const PixelReader::PixelData* pix,
			      std::array<Label,Cluster::maxLabels> &labels,
			      int &nfilled) const
{
  // transfer MC labels to cluster
  if (nfilled>=Cluster::maxLabels) return;
  for (int id=0;id<Digit::maxLabels;id++) {
    Label lbl = pix->labels[id];
    if ( lbl.isEmpty() ) return; // all following labels will be invalid
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
