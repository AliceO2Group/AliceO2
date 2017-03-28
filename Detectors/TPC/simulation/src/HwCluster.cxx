/// \file HwCluster.cxx
/// \brief Class to have some more info about the HwClusterer clusters

#include "TPCSimulation/HwCluster.h"
//#include "TPCSimulation/Cluster.h"
#include "FairLogger.h"
#include "TMath.h"

ClassImp(AliceO2::TPC::HwCluster)

using namespace AliceO2::TPC;

//________________________________________________________________________
HwCluster::HwCluster(Short_t sizeP, Short_t sizeT, UShort_t fpTotalWidth, UShort_t fpDecPrec):
  Cluster(),
  mPad(-1),
  mTime(-1),
  mSizeP(sizeP),
  mSizeT(sizeT),
  mSize(0),
  mFixedPointTotalWidth(fpTotalWidth),
  mFixedPointDecPrec(fpDecPrec),
  mPropertiesCalculatd(kFALSE)
{
  mClusterData = new HwFixedPoint*[mSizeT];
  for (Int_t t=0; t<mSizeT; t++){
    mClusterData[t] = new HwFixedPoint[mSizeP];
    for (Int_t p=0; p<mSizeP; p++){
      mClusterData[t][p] = HwFixedPoint(0,mFixedPointTotalWidth,mFixedPointDecPrec);
    }
  }
}

//________________________________________________________________________
HwCluster::HwCluster(Short_t cru, Short_t row, Short_t sizeP, Short_t sizeT, 
    Float_t** clusterData, Short_t maxPad, Short_t maxTime,
    UShort_t fpTotalWidth, UShort_t fpDecPrec):
  Cluster(cru,row,-1,-1,-1,-1,-1,-1),
  mPad(maxPad),
  mTime(maxTime),
  mSizeP(sizeP),
  mSizeT(sizeT),
  mSize(0),
  mFixedPointTotalWidth(fpTotalWidth),
  mFixedPointDecPrec(fpDecPrec),
  mPropertiesCalculatd(kFALSE)
{
  mClusterData = new HwFixedPoint*[mSizeT];
  for (Int_t t=0; t<mSizeT; t++){
    mClusterData[t] = new HwFixedPoint[mSizeP];
    for (Int_t p=0; p<mSizeP; p++){
      mClusterData[t][p] = HwFixedPoint(clusterData[t][p],mFixedPointTotalWidth,mFixedPointDecPrec);
      if (TMath::Abs(clusterData[t][p] - (Float_t) mClusterData[t][p]) > 0.001) std::cout << clusterData[t][p] << " " << mClusterData[t][p] << " " << TMath::Abs(clusterData[t][p] - (Float_t) mClusterData[t][p]) << std::endl;
    }
  }

  calculateClusterProperties(cru,row);
}

//________________________________________________________________________
HwCluster::HwCluster(const HwCluster& other):
  Cluster(other),
  mPad(other.mPad),
  mTime(other.mTime),
  mSizeP(other.mSizeP),
  mSizeT(other.mSizeT),
  mSize(other.mSize),
  mFixedPointTotalWidth(other.mFixedPointTotalWidth),
  mFixedPointDecPrec(other.mFixedPointDecPrec),
  mPropertiesCalculatd(other.mPropertiesCalculatd)
{
  mClusterData = new HwFixedPoint*[mSizeT];
  for (Int_t t=0; t<mSizeT; t++){
    mClusterData[t] = new HwFixedPoint[mSizeP];
    for (Int_t p=0; p<mSizeP; p++){
      mClusterData[t][p] = other.mClusterData[t][p];
    }
  }
}

//________________________________________________________________________
HwCluster::~HwCluster()
{
  for (Int_t t=0; t<mSizeT; t++){
//    for (Int_t p=0; p<mSizeP; p++){
//      delete mClusterData[t][p];
//    }
    delete [] mClusterData[t];
  }
  delete [] mClusterData;
}

//________________________________________________________________________
void HwCluster::setClusterData(Short_t cru, Short_t row, Short_t sizeP, Short_t sizeT, 
    Float_t** clusterData, Short_t maxPad, Short_t maxTime)
{
  if (sizeP != mSizeP || sizeT != mSizeT) {
    LOG(ERROR) << "Given cluster size does not match. Abort..." << FairLogger::endl;
  }
  for (Int_t t=0; t<mSizeT; t++){
    for (Int_t p=0; p<mSizeP; p++){
//      delete ClusterData[t][p];
      mClusterData[t][p] = HwFixedPoint(clusterData[t][p],mFixedPointTotalWidth,mFixedPointDecPrec);
    }
  }
  mPad = maxPad;
  mTime = maxTime;

  calculateClusterProperties(cru, row);
}

//________________________________________________________________________
void HwCluster::calculateClusterProperties(Short_t cru, Short_t row)
{
  if (mPropertiesCalculatd) return;

  HwFixedPoint qMax(mClusterData[2][2],mFixedPointTotalWidth,mFixedPointDecPrec);   // central pad
  HwFixedPoint qTot(0,mFixedPointTotalWidth+5,mFixedPointDecPrec);//qMax;
  HwFixedPoint charge(0,mFixedPointTotalWidth+5,mFixedPointDecPrec);
  HwFixedPoint meanP(0,mFixedPointTotalWidth+9,mFixedPointDecPrec+3);
  HwFixedPoint meanT(0,mFixedPointTotalWidth+9,mFixedPointDecPrec+3);
  //Double_t qMax = mClusterData[2][2];   // central pad
  //Double_t qTot = 0;//qMax;
  //Double_t charge = 0;
  //Double_t meanP = 0;
  //Double_t meanT = 0;
  Double_t sigmaP = 0;
  Double_t sigmaT = 0;
  Short_t minT = mSizeT;
  Short_t maxT = 0;
  Short_t minP = mSizeP;
  Short_t maxP = 0;

  for (Short_t t = 0; t < mSizeT; t++) {
    Int_t deltaT = t - mSizeT/2;
    for (Short_t p = 0; p < mSizeP; p++) {
      Int_t deltaP = p - mSizeP/2;

      charge = mClusterData[t][p];

      qTot += charge;

      meanP += charge * (p+1);//deltaP;
      meanT += charge * (t+1);//deltaT;

      sigmaP += (Double_t) charge * deltaP*deltaP;
      sigmaT += (Double_t) charge * deltaT*deltaT;

      if (charge > 0) {
        minP = TMath::Min(minP,p); maxP = TMath::Max(maxP,p);
        minT = TMath::Min(minT,t); maxT = TMath::Max(maxT,t);
      }
      
    }
  }

  mSize = (maxP-minP+1)*10 + (maxT-minT+1);

  if (qTot > 0) {
    meanP  /= qTot;
    meanP -= 3;
    meanT  /= qTot;
    meanT -= 3;
    sigmaP /= (Double_t) qTot;
    sigmaT /= (Double_t) qTot;

    sigmaP = TMath::Sqrt(sigmaP - ((Double_t)meanP*(Double_t)meanP));
    sigmaT = TMath::Sqrt(sigmaT - ((Double_t)meanT*(Double_t)meanT));

    meanP += mPad;
    meanT += mTime;
  }

  setParameters(cru,row,(Double_t)qTot,(Double_t)qMax,(Double_t)meanP,sigmaP,(Double_t)meanT,sigmaT);

  mPropertiesCalculatd = kTRUE;
}

//________________________________________________________________________
std::ostream& HwCluster::Print(std::ostream &output) const
{
  Cluster::Print(output);
  output << " centered at (pad, time) = " << mPad << ", " << mTime
    << " covering " << Int_t(mSize/10)  << " pads and " << mSize%10
    << " time bins";
  return output;
}

//________________________________________________________________________
std::ostream& HwCluster::PrintDetails(std::ostream &output) const
{
  Cluster::Print(output);
  output << " centered at (pad, time) = " << mPad << ", " << mTime
    << " covering " << Int_t(mSize/10)  << " pads and " << mSize%10
    << " time bins" << " " << std::endl;
  for (Int_t t=0; t<mSizeT; t++){
    for (Int_t p=0; p<mSizeP; p++){
      output << "\t" << mClusterData[t][p];
    }
    output << std::endl;
  }
//  output << std::endl;

  return output;
}

