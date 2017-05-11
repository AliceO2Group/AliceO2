/// \file   SimpleEventDisplay.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TH1D.h"
#include "TH2S.h"
#include "TROOT.h"
#include "TString.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"

#include "TPCMonitor/SimpleEventDisplay.h"

using namespace o2::TPC;

SimpleEventDisplay::SimpleEventDisplay()
  : CalibRawBase(),
    mHnDataIROC(nullptr),
    mHnDataOROC(nullptr),
    mPadMax(PadSubset::ROC),
    mHSigIROC(nullptr),
    mHSigOROC(nullptr),
    mPedestals(nullptr),
    mCurrentChannel(-1),
    mCurrentROC(-1),
    mLastSector(-1),
    mSelectedSector(-1),
    mLastSelSector(-1),
    mCurrentRow(-1),
    mCurrentPad(-1),
    mMaxPadSignal(-1),
    mMaxTimeBin(-1),
    mSectorLoop(kFALSE),
    mFirstTimeBin(0),
    mLastTimeBin(512),
    mTPCmapper(Mapper::instance())
{
  //
  //
  //
  const int nPadsIROC = mTPCmapper.getPadsInIROC();
  const int nPadsOROC = mTPCmapper.getPadsInOROC();
  const int numberOfTimeBins = mLastTimeBin - mFirstTimeBin;

  Int_t binsIROC[3] = {36, nPadsIROC, numberOfTimeBins};
  Double_t xminIROC[3] = {0., 0, Double_t(mFirstTimeBin)};
  Double_t xmaxIROC[3] = {36., Double_t(nPadsIROC), Double_t(mLastTimeBin)};
  Int_t binsOROC[3] = {36, nPadsOROC, 1000};
  Double_t xminOROC[3] = {0.,0., Double_t(mFirstTimeBin)};
  Double_t xmaxOROC[3] = {36., Double_t(nPadsOROC), Double_t(mLastTimeBin)};
  mHnDataIROC = new THnSparseS("mHnDataIROC","IROC data", 3, binsIROC, xminIROC, xmaxIROC);
  mHnDataOROC = new THnSparseS("mHnDataOROC","OROC data", 3, binsOROC, xminOROC, xmaxOROC);
  mHSigIROC = new TH2D("PadSigIROC","Pad Signals IROC", numberOfTimeBins, mFirstTimeBin, mLastTimeBin, nPadsIROC, 0, nPadsIROC);
  mHSigOROC = new TH2D("PadSigOROC","Pad Signals IROC", numberOfTimeBins, mFirstTimeBin, mLastTimeBin, nPadsOROC, 0, nPadsOROC);
  //printf("Selected ini: %d %p\n",0, mPadMax.GetCalROC(0));
}

//_____________________________________________________________________
Int_t SimpleEventDisplay::Update(const Int_t roc,
                                 const Int_t row,
                                 const Int_t pad,
                                 const Int_t timeBin,
                                 const Float_t signal)
{
  //
  // Signal filling methode on the fly pedestal and time offset correction if necessary.
  // no extra analysis necessary. Assumes knowledge of the signal shape!
  // assumes that it is looped over consecutive time bins of one pad
  //
  //printf("Update called: %d, %d, %d, %d, %.3f\n", roc, row, pad, timeBin, signal);
  if (row<0) return 0;
  if (pad<0) return 0;
  if (timeBin<0) return 0;
  if ( (timeBin>mLastTimeBin) || (timeBin<mFirstTimeBin)   ) return 0;
  if (mSectorLoop && roc%36!=mSelectedSector%36) return 0;

  if ( row<0 || pad<0 ){
    printf("Wrong Pad or Row number, skipping!");
    return 0;
  }

  float corrSignal = signal;

  // ===| get pedestal |========================================================
  if (mPedestals) {
    corrSignal -= mPedestals->getValue(ROC(roc), row, pad);
  }

  Int_t iChannel  = mTPCmapper.globalPadNumber(PadPos(row, pad));

    //init first pad and roc in this event
  if ( mCurrentChannel == -1 ) {
    mCurrentChannel = iChannel;
    mCurrentROC  = roc;
    mCurrentRow     = row;
    mCurrentPad     = pad;
  }

    //process last pad if we change to a new one
  if ( iChannel != mCurrentChannel ){
    //ProcessPad();
    mLastSector=mCurrentROC;
    mCurrentChannel = iChannel;
    mCurrentROC  = roc;
    mCurrentRow     = row;
    mCurrentPad     = pad;
    mMaxPadSignal=0;
  }
//  if (signal>0) printf("%02d:%03d:%03d:%05d: %.3f\n",mCurrentROC,mCurrentRow,mCurrentPad,mCurrentChannel,signal);
  //fill signals for current pad
  if (mCurrentROC%36==mSelectedSector%36){
    const Int_t nbins = mLastTimeBin - mFirstTimeBin;
    const Int_t offset = (nbins+2)*(iChannel+1)+timeBin+1;

    if ((UInt_t)roc<mTPCmapper.getNumberOfIROCs()){
      mHSigIROC->GetArray()[offset]=corrSignal;
    }else{
      mHSigOROC->GetArray()[offset]=corrSignal;
    }
  }

  CalROC& calROC = mPadMax.getCalArray(mCurrentROC);
  //auto val = calROC.getValue(mCurrentRow, mCurrentPad);
  auto val = calROC.getValue(row, pad);

  if ( corrSignal > val ){
    //printf("sec: %2d, row: %2d, pad: %3d, val: %.2f, sig: (%.2f) %.2f\n", mCurrentROC, mCurrentRow, mCurrentPad, val, corrSignal, signal);
    //calROC.setValue(mCurrentRow,mCurrentPad,signal);
    calROC.setValue(row, pad, corrSignal);
    mMaxPadSignal = corrSignal;
    mMaxTimeBin   = timeBin;
  }
  //printf("Update done\n");
  return 0;
}

//_____________________________________________________________________
TH1D* SimpleEventDisplay::MakePadSignals(Int_t roc, Int_t row, Int_t pad)
{
  // TODO: check
  //if (roc<0||roc>=(Int_t)mROC->GetNSectors()) return nullptr;
  //if (row<0||row>=(Int_t)mROC->GetNRows(roc)) return nullptr;
  //if (pad<0||pad>=(Int_t)mROC->GetNPads(roc,row)) return nullptr;
  // TODO: possible bug for OROC
  const Int_t channel =  
    mTPCmapper.globalPadNumber(PadPos(row,pad))
    - (roc>=mTPCmapper.getNumberOfIROCs()) * mTPCmapper.getPadsInIROC();

  mSelectedSector=roc;
//  mLastSelSector=roc;
  //attention change for if event has changed
  if (mSelectedSector%36!=mLastSelSector%36){
    mSectorLoop=kTRUE;
    ProcessEvent();
    mLastSelSector=mSelectedSector;
    mSectorLoop=kFALSE;
  }
  TH1D *h=nullptr;
  const Int_t nbins = mLastTimeBin - mFirstTimeBin;
  if (nbins<=0) return nullptr;
  const Int_t offset = (nbins+2)*(channel+1);
  Double_t *arrP=nullptr;

  const FECInfo& fecInfo = mTPCmapper.getFECInfo(PadROCPos(roc, row, pad));

  TString title("#splitline{#lower[.1]{#scale[.5]{");

  if (roc<(Int_t)mTPCmapper.getNumberOfIROCs()){
    h=(TH1D*)gROOT->FindObject("PadSignals_IROC");
    if (!h) h=new TH1D("PadSignals_IROC","PadSignals_IROC",nbins,mFirstTimeBin,mLastTimeBin);
    h->SetFillColor(kBlue-10);
    arrP=mHSigIROC->GetArray()+offset;
//     title+="IROC ";
  } else {
    h=(TH1D*)gROOT->FindObject("PadSignals_OROC");
    if (!h) h=new TH1D("PadSignals_OROC","PadSignals_OROC",nbins,mFirstTimeBin,mLastTimeBin);
    h->SetFillColor(kBlue-10);
    arrP=mHSigOROC->GetArray()+offset;
    title+="OROC ";
  }
  title+=(roc/18%2==0)?"A":"C";
  title+=Form("%02d (%02d) row: %02d, pad: %03d, globalpad: %05d (in roc)}}}{#scale[.5]{FEC: %02d, Chip: %02d, Chn: %02d}}",
              roc%18,roc,row,pad,channel, fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel());
  //title+=Form("row: %02d, pad: %03d, cpad: %03d, globalpad: %05d}}}{#scale[.5]{br: %d, FEC: %02d, Chip: %02d, Chn: %02d = HW: %d}}",
              //row,pad,pad-mROC->GetNPads(roc,row)/2,channel,
              //mTPCmapper.GetBranch(roc,row,pad),
              //mTPCmapper.GetFEChw(roc,row,pad),
              //mTPCmapper.GetChip(roc,row,pad),
              //mTPCmapper.GetChannel(roc,row,pad),
              //mTPCmapper.GetHWAddress(roc,row,pad)
            //);
  h->SetTitle(title.Data());
  Int_t entries=0;
  for ( Int_t i=0; i<nbins; i++ ) entries+=(Int_t)arrP[i+1];
  h->Set(nbins+2,arrP);
  h->SetEntries(entries);
  return h;
}
//_____________________________________________________________________
void SimpleEventDisplay::ResetEvent()
{
  //
  //
  //
  //for (auto reader : mGBTFrameContainers) {
    //reader->reProcessAllFrames();
  //}
  mHnDataIROC->Reset();
  mHnDataOROC->Reset();
  if (!mSectorLoop) mPadMax.multiply(0.);
  mHSigIROC->Reset();
  mHSigOROC->Reset();
}

