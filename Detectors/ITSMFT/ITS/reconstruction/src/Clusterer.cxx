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

  mPreClusters.clear();
  mIndices.clear();
  mLabels.clear();

  mChipID=chipID;
  mCol=col;
  mCurr[row]=0;
  mPreClusters.emplace_back(1,std::pair<UShort_t,UShort_t>(row,col)); //start the first precluster
  mIndices.push_back(0);
  mLabels.push_back(label);
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
     auto &ci = mIndices[pci];
     if (attached) {
        auto &newci = mIndices[mCurr[row]];
	if (ci < newci) newci = ci;
	else ci = newci;
     } else {
        mPreClusters[ci].emplace_back(row,col);
	mCurr[row] = pci;
	attached = true;
     }
  }
  
  if (attached) return;

  //start new precluster
  mPreClusters.emplace_back(1,std::pair<UShort_t,UShort_t>(row,col));
  Int_t lastIndex = mPreClusters.size()-1;
  mCurr[row] = lastIndex;
  mIndices.push_back(lastIndex);
  mLabels.push_back(label);

}

void Clusterer::finishChip(TClonesArray &clusters)
{
  static Float_t sigmaX2 = mPitchX * mPitchX / 12.; //FIXME
  static Float_t sigmaY2 = mPitchZ * mPitchZ / 12.;

  for (Int_t i1=0; i1<mPreClusters.size(); ++i1) {
    const auto &preCluster1 = mPreClusters[i1];
    const auto ci = mIndices[i1];
    if (ci<0) continue;
    UShort_t xmax=0, xmin=65535;
    UShort_t zmax=0, zmin=65535;
    Float_t x=0., z=0.;
    Int_t npix = preCluster1.size();
    for (const auto &dig : preCluster1) {
      x += dig.first;
      z += dig.second;
      if (dig.first < xmin) xmin=dig.first;
      if (dig.first > xmax) xmax=dig.first;
      if (dig.second < zmin) zmin=dig.second;
      if (dig.second > zmax) zmax=dig.second;
    }
    mIndices[i1] = -1;
    for (Int_t i2=i1+1; i2<mPreClusters.size(); ++i2) {
      const auto &preCluster2 = mPreClusters[i2];
      if (mIndices[i2] != ci) continue;
      npix += preCluster2.size();
      for (const auto &dig : preCluster2) {
        x += dig.first;
        z += dig.second;
        if (dig.first < xmin) xmin=dig.first;
        if (dig.first > xmax) xmax=dig.first;
        if (dig.second < zmin) zmin=dig.second;
        if (dig.second > zmax) zmax=dig.second;
      }
      mIndices[i2] = -1;
    }    
    x /= npix;
    x = mX0 + x*mPitchX;
    z /= npix;
    z = mZ0 + z*mPitchZ;
    Cluster c;
    c.setVolumeId(mChipID);
    c.setX(x);
    c.setY(0);
    c.setZ(z);
    c.setSigmaY2(sigmaX2);
    c.setSigmaZ2(sigmaY2);
    c.setNxNzN(xmax-xmin+1,zmax-zmin+1,npix);
    c.setFrameLoc();
    c.setLabel(mLabels[i1], 0);
    new (clusters[clusters.GetEntriesFast()]) Cluster(c);
  }
}
