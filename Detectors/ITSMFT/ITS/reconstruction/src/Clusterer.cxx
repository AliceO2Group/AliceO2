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
  :mCurr(mColumn2)
  ,mPrev(mColumn1)
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
  mPrev=mColumn1;
  mCurr=mColumn2;
  std::fill(std::begin(mColumn1), std::end(mColumn1), -1);
  std::fill(std::begin(mColumn2), std::end(mColumn2), -1);

  mPreClusters.clear();
  mLabels.clear();

  mChipID=chipID;
  mCol=col;
  mCurr[row+1]=0;
  mPreClusters.emplace_back(1,std::pair<UShort_t,UShort_t>(row,col)); //start the first precluster
  mLabels.push_back(label);
}

void Clusterer::updateChip(UShort_t chipID, UShort_t row, UShort_t col, Int_t label)
{
  if (mCol != col) { // switch the buffers
     Int_t *tmp=mCurr;
     mCurr=mPrev;
     mPrev=tmp;
     if (col > mCol+1) std::fill(mPrev, mPrev+kMaxRow+2, -1);
     std::fill(mCurr, mCurr+kMaxRow+2, -1);
     mCol=col;
  }

  Int_t idx=row+1;
  auto clusterIdx = mCurr[idx-1];     //upper
  if (clusterIdx < 0) {
     clusterIdx = mPrev[idx];         //left
     if (clusterIdx < 0) {
        clusterIdx = mPrev[idx-1];    //upper left
        if (clusterIdx < 0) {
	   clusterIdx = mPrev[idx+1]; //lower left
	   if (clusterIdx < 0) {
	      mCurr[idx]=mPreClusters.size();
	      mPreClusters.emplace_back(1,std::pair<UShort_t,UShort_t>(row,col)); //new precluster
              mLabels.push_back(label);
              return;
	   }
        }
     }
  }
  mCurr[idx] = clusterIdx;
  mPreClusters[clusterIdx].emplace_back(row,col); //update an existing precluster
}

void Clusterer::finishChip(TClonesArray &clusters)
{
  static Float_t sigmaX2 = mPitchX * mPitchX / 12.; //FIXME
  static Float_t sigmaY2 = mPitchZ * mPitchZ / 12.;

  Int_t i=0;
  for (auto pre : mPreClusters) {
    Int_t npix = pre.size();
    Float_t x=0., z=0.;
    for (auto dig : pre) {
      x += dig.first;
      z += dig.second;
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
    c.setNxNzN(3,3,npix); //FIXME
    c.setFrameLoc();
    c.setLabel(mLabels[i++], 0);
    new (clusters[clusters.GetEntriesFast()]) Cluster(c);
  }
}
