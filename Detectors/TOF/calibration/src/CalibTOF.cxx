// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <TTree.h>
#include <cassert>

#include <fairlogger/Logger.h>
#include "TOFBase/Geo.h"

#include <TFile.h>
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/PID.h"

#include "TOFCalibration/CalibTOF.h"

#include "CommonConstants/LHCConstants.h"

#include "TMath.h"
#include "TRandom.h"

using namespace o2::globaltracking;

ClassImp(CalibTOF);

//______________________________________________
CalibTOF::CalibTOF()
{

  // constructor needed to instantiate the pointers of the class (histos + array)

  for (int ipad = 0; ipad < NPADSPERSTEP; ipad++) {
  }
  mHistoChTimeSlewingAll = new TH2F("hTOFchTimeSlewingAll", ";tot (ns);t - t_{exp} - t_{offset} (ps)", 5000, 0., 250., 1000, -24400., 24400.);
}
//______________________________________________
CalibTOF::~CalibTOF()
{

  // destructor

  if (mHistoLHCphase) {
    delete mHistoLHCphase;
  }
  delete mHistoChTimeSlewingAll;
}
//______________________________________________
void CalibTOF::attachInputTrees()
{
  ///< attaching the input tree

  if (!mTreeCollectedCalibInfoTOF) {
    LOG(fatal) << "Input tree with collected TOF calib infos is not set";
  }

  if (!mTreeCollectedCalibInfoTOF->GetBranch(mCollectedCalibInfoTOFBranchName.data())) {
    LOG(fatal) << "Did not find collected TOF calib info branch " << mCollectedCalibInfoTOFBranchName << " in the input tree";
  }
  /*
  LOG(info) << "Attached tracksTOF calib info " << mCollectedCalibInfoTOFBranchName << " branch with " << mTreeCollectedCalibInfoTOF->GetEntries()
            << " entries";
  */
}

//______________________________________________
void CalibTOF::init()
{
  ///< initizalizations

  if (mInitDone) {
    LOG(error) << "Initialization was already done";
    return;
  }

  TH1::AddDirectory(0); // needed because we have the LHCPhase created here, while in the macro we might have the output file open
                        // (we don't want to bind the histogram to the file, or teh destructor will complain)

  attachInputTrees();

  std::fill_n(mCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);
  std::fill_n(mCalibChannelOffsetErr, o2::tof::Geo::NCHANNELS, -1);
  std::fill_n(mInitialCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);

  // this is only to test random offsets!!!!
  for (int i = 0; i < o2::tof::Geo::NCHANNELS; i++) {
    mInitialCalibChannelOffset[i] = gRandom->Rndm() * 25000 - 12500;
  }

  // load better knoldge of channel offset (from CCDB?)
  // to be done

  // create output branch with output -- for now this is empty
  if (mOutputTree) {
    mLHCphaseObj = new o2::dataformats::CalibLHCphaseTOF();
    mTimeSlewingObj = new o2::dataformats::CalibTimeSlewingParamTOF();
    mOutputTree->Branch("mLHCphaseObj", &mLHCphaseObj);
    mOutputTree->Branch("mTimeSlewingObj", &mTimeSlewingObj);

    //    LOG(info) << "Matched tracks will be stored in " << mOutputBranchName << " branch of tree "
    //              << mOutputTree->GetName();
  } else {
    LOG(error) << "Output tree is not attached, matched tracks will not be stored";
  }

  // booking the histogram of the LHCphase
  int nbinsLHCphase = TMath::Min(1000, int((mMaxTimestamp - mMinTimestamp) / 300) + 1);
  if (nbinsLHCphase < 1000) {
    mMaxTimestamp = mMinTimestamp + nbinsLHCphase * 300; // we want that the last bin of the histogram is also large 300s; this we need to do only when we have less than 1000 bins, because in this case we will integrate over intervals that are larger than 300s anyway
  }
  mHistoLHCphase = new TH2F("hLHCphase", ";clock offset (ps); timestamp (s)", 1000, -24400, 24400, nbinsLHCphase, mMinTimestamp, mMaxTimestamp);

  // setting CCDB for output
  mCalibTOFapi.setURL(mCCDBpath.c_str());

  mInitDone = true;

  print();
}
//______________________________________________
void CalibTOF::run(int flag, int sector)
{
  ///< running the matching

  Int_t currTOFInfoTreeEntry = -1;

  std::vector<o2::dataformats::CalibInfoTOFshort>* localCalibInfoTOF = nullptr;
  TFile fOpenLocally(mTreeCollectedCalibInfoTOF->GetCurrentFile()->GetName());
  TTree* localTree = (TTree*)fOpenLocally.Get(mTreeCollectedCalibInfoTOF->GetName());

  if (!localTree) {
    LOG(fatal) << "tree " << mTreeCollectedCalibInfoTOF->GetName() << " not found in " << mTreeCollectedCalibInfoTOF->GetCurrentFile()->GetName();
  }

  localTree->SetBranchAddress(mCollectedCalibInfoTOFBranchName.data(), &localCalibInfoTOF);

  if (!mInitDone) {
    LOG(fatal) << "init() was not done yet";
  }

  TStopwatch timerTot;
  timerTot.Start();

  if (flag & kLHCphase) {                                                // LHC phase --> we will use all the entries in the tree
    while (loadTOFCollectedCalibInfo(localTree, currTOFInfoTreeEntry)) { // fill here all histos you need
      fillLHCphaseCalibInput(localCalibInfoTOF);                         // we will fill the input for the LHC phase calibration
    }
    doLHCPhaseCalib();
  }
  // channel offset + problematic (flag = 2), or time slewing (flag = 4)
  if ((flag & kChannelOffset) || (flag & kChannelTimeSlewing)) { // for the moment compute everything idependetly of the flag
    TH1F* histoChOffsetTemp[NPADSPERSTEP];
    std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad[NPADSPERSTEP];
    for (int ipad = 0; ipad < NPADSPERSTEP; ipad++) {
      histoChOffsetTemp[ipad] = new TH1F(Form("OffsetTemp_Sec%02d_Pad%04d", sector, ipad), Form("Sector %02d (pad = %04d);channel offset (ps)", sector, ipad), 1000, -24400, 24400);
      if (flag & kChannelTimeSlewing) {
        calibTimePad[ipad] = new std::vector<o2::dataformats::CalibInfoTOFshort>; // temporary array containing [time, tot] for every pad that we process; this will be the input for the 2D histo for timeSlewing calibration (to be filled after we get the channel offset)
      } else {
        calibTimePad[ipad] = nullptr;
      }
    }

    TF1* funcChOffset = new TF1(Form("fTOFchOffset_%02d", sector), "[0]*TMath::Gaus((x-[1])*(x-[1] < 12500 && x-[1] > -12500) + (x-[1]+25000)*(x-[1] < -12500) + (x-[1]-25000)*(x-[1] > 12500),0,[2])*(x > -12500 && x < 12500)", -12500, 12500);
    funcChOffset->SetParLimits(1, -12500, 12500);
    funcChOffset->SetParLimits(2, 50, 2000);

    TH2F* histoChTimeSlewingTemp = new TH2F(Form("hTOFchTimeSlewingTemp_%02d", sector), Form("Sector %02d;tot (ns);t - t_{exp} - t_{offset} (ps)", sector), 5000, 0., 250., 1000, -24400., 24400.);

    int startLoop = 0;                     // first pad that we will process in this process (we are processing a sector, unless sector = -1)
    int endLoop = o2::tof::Geo::NCHANNELS; // last pad that we will process in this process (we are processing a sector)
    if (sector > -1) {
      startLoop = sector * o2::tof::Geo::NPADSXSECTOR;  // first pad that we will process in this process (we are processing a sector, unless sector = -1)
      endLoop = startLoop + o2::tof::Geo::NPADSXSECTOR; // last pad that we will process in this process (we are processing a sector, unless sector = -1)
    }
    for (int ich = startLoop; ich < endLoop; ich += NPADSPERSTEP) {
      sector = ich / o2::tof::Geo::NPADSXSECTOR; // we change the value of sector which is needed when it is "-1" to put
                                                 // in the output histograms a meaningful name; this is not needed in
                                                 // case we run with sector != -1, but it will not hurt :)
      resetChannelLevelHistos(histoChOffsetTemp, histoChTimeSlewingTemp, calibTimePad);
      printf("strip %i\n", ich / 96);
      currTOFInfoTreeEntry = ich - 1;
      int ipad = 0;
      int entryNext = currTOFInfoTreeEntry + o2::tof::Geo::NCHANNELS;

      while (loadTOFCollectedCalibInfo(localTree, currTOFInfoTreeEntry)) { // fill here all histos you need

        histoChOffsetTemp[ipad]->SetName(Form("OffsetTemp_Sec%02d_Pad%04d", sector, (ipad + ich) % o2::tof::Geo::NPADSXSECTOR));
        fillChannelCalibInput(localCalibInfoTOF, mInitialCalibChannelOffset[ich + ipad], ipad, histoChOffsetTemp[ipad], calibTimePad[ipad]); // we will fill the input for the channel-level calibration
        ipad++;

        if (ipad == NPADSPERSTEP) {
          ipad = 0;
          currTOFInfoTreeEntry = entryNext;
          entryNext += o2::tof::Geo::NCHANNELS;
        }
      }
      TFile* fout = nullptr;
      if (flag & kChannelTimeSlewing && mDebugMode) {
        fout = new TFile(Form("timeslewingTOF%06i.root", ich / 96), "RECREATE");
      }

      for (ipad = 0; ipad < NPADSPERSTEP; ipad++) {
        if (histoChOffsetTemp[ipad]->GetEntries() > 30) {
          float fractionUnderPeak = doChannelCalibration(ipad, histoChOffsetTemp[ipad], funcChOffset);
          mCalibChannelOffset[ich + ipad] = funcChOffset->GetParameter(1) + mInitialCalibChannelOffset[ich + ipad];

          int channelInSector = (ipad + ich) % o2::tof::Geo::NPADSXSECTOR;

          mTimeSlewingObj->setFractionUnderPeak(sector, channelInSector, fractionUnderPeak);
          mTimeSlewingObj->setSigmaPeak(sector, channelInSector, abs(funcChOffset->GetParameter(2)));

          // now fill 2D histo for time-slewing using current channel offset

          if (flag & kChannelTimeSlewing) {
            histoChTimeSlewingTemp->Reset();
            fillChannelTimeSlewingCalib(mCalibChannelOffset[ich + ipad], ipad, histoChTimeSlewingTemp, calibTimePad[ipad]); // we will fill the input for the channel-time-slewing calibration

            histoChTimeSlewingTemp->SetName(Form("TimeSlewing_Sec%02d_Pad%04d", sector, channelInSector));
            histoChTimeSlewingTemp->SetTitle(Form("Sector %02d (pad = %04d)", sector, channelInSector));
            TGraphErrors* gTimeVsTot = processSlewing(histoChTimeSlewingTemp, 1, funcChOffset);

            if (gTimeVsTot && gTimeVsTot->GetN()) {
              for (int itot = 0; itot < gTimeVsTot->GetN(); itot++) {
                mTimeSlewingObj->addTimeSlewingInfo(ich + ipad, gTimeVsTot->GetX()[itot], gTimeVsTot->GetY()[itot] + mCalibChannelOffset[ich + ipad]);
              }
            } else { // just add the channel offset
              mTimeSlewingObj->addTimeSlewingInfo(ich + ipad, 0, mCalibChannelOffset[ich + ipad]);
            }

            if (mDebugMode && gTimeVsTot && gTimeVsTot->GetN() && fout) {
              fout->cd();
              int istrip = ((ich + ipad) / o2::tof::Geo::NPADS) % o2::tof::Geo::NSTRIPXSECTOR;
              gTimeVsTot->SetName(Form("pad_%02d_%02d_%02d", sector, istrip, ipad % o2::tof::Geo::NPADS));
              gTimeVsTot->Write();
              //	      histoChTimeSlewingTemp->Write(Form("histoChTimeSlewingTemp_%02d_%02d_%02d", sector, istrip, ipad%o2::tof::Geo::NPADS)); // no longer written since it produces a very large output
            }
          } else if (flag & kChannelOffset) {
            mTimeSlewingObj->addTimeSlewingInfo(ich + ipad, 0, mCalibChannelOffset[ich + ipad]);
          }
        }
      }
      if (fout) {
        fout->Close();
        delete fout;
      }
    }

    for (int ipad = 0; ipad < NPADSPERSTEP; ipad++) {
      delete histoChOffsetTemp[ipad];
      if (calibTimePad[ipad]) {
        delete calibTimePad[ipad];
      }
    }
    delete histoChTimeSlewingTemp;
    delete funcChOffset;
  }

  fOpenLocally.Close();

  timerTot.Stop();
  printf("Timing (%i):\n", sector);
  printf("Total:        ");
  timerTot.Print();
}

//______________________________________________
void CalibTOF::fillOutput(int flag)
{
  mOutputTree->Fill();

  if (mFillCCDB) {
    if (flag & kLHCphase) {
      std::map<std::string, std::string> metadataLHCphase;                                                                                  // can be empty
      mCalibTOFapi.writeLHCphase(mLHCphaseObj, metadataLHCphase, (uint64_t)mMinTimestamp * 1000, (uint64_t)mMaxTimestamp * 1000);           // we use as validity the timestamps that we got from the input for the calibration; but we need to convert to ms for the CCDB (at least for now that we use an integer for the timestamp)
    }
    if (flag & kChannelOffset || flag & kChannelTimeSlewing) {
      std::map<std::string, std::string> metadataChannelCalib;                                                        // can be empty
      mCalibTOFapi.writeTimeSlewingParam(mTimeSlewingObj, metadataChannelCalib, (uint64_t)mMinTimestamp * 1000);      // contains both offset and time slewing; we use as validity the START ONLY timestamp that we got from the input for the calibration; but we need to convert to ms for the CCDB (at least for now that we use an integer for the timestamp), END is default
    }
  }
}

//______________________________________________
void CalibTOF::print() const
{
  ///< print the settings

  LOG(info) << "****** component for calibration of TOF channels ******";
  if (!mInitDone) {
    LOG(info) << "init is not done yet - nothing to print";
    return;
  }

  LOG(info) << "**********************************************************************";
}

//______________________________________________
bool CalibTOF::loadTOFCollectedCalibInfo(TTree* localTree, int& currententry, int increment)
{
  ///< load next chunk of TOF infos
  //  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeCollectedCalibInfoTOF->GetEntries());

  currententry += increment;
  while (currententry < localTree->GetEntries()) {
    //while (currententry < 800000){
    //    && currententry < o2::tof::Geo::NCHANNELS) {
    localTree->GetEntry(currententry);
    //LOG(info) << "Loading TOF calib info entry " << currententry << " -> " << mCalibInfoTOF->size()<< " infos";

    return true;
  }
  currententry -= increment;

  return false;
}

//______________________________________________

void CalibTOF::fillLHCphaseCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof)
{

  // we will fill the input for the LHC phase calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1. / bc;

  for (auto infotof = calibinfotof->begin(); infotof != calibinfotof->end(); infotof++) {
    double dtime = infotof->getDeltaTimePi();
    dtime -= (int(dtime * bc_inv + 5.5) - 5) * bc; // do truncation far (by 5 units) from zero to avoid truncation of negative numbers

    mHistoLHCphase->Fill(dtime, infotof->getTimestamp());
  }
}
//______________________________________________

void CalibTOF::doLHCPhaseCalib()
{

  // calibrate with respect LHC phase

  if (!mFuncLHCphase) {
    mFuncLHCphase = new TF1("fLHCphase", "gaus");
  }

  int ifit0 = 1;
  for (int ifit = ifit0; ifit <= mHistoLHCphase->GetNbinsY(); ifit++) {
    TH1D* htemp = mHistoLHCphase->ProjectionX("htemp", ifit0, ifit);
    if (htemp->GetEntries() < 300) {
      // we cannot fit the histogram, we will merge with the next bin
      //      Printf("We don't have enough entries to fit");
      continue;
    }

    int res = FitPeak(mFuncLHCphase, htemp, 500., 3., 2., "LHCphase");
    if (res) {
      continue;
    }

    mLHCphaseObj->addLHCphase(mHistoLHCphase->GetYaxis()->GetBinLowEdge(ifit0), mFuncLHCphase->GetParameter(1));
    ifit0 = ifit + 1; // starting point for the next LHC interval
  }
}
//______________________________________________

void CalibTOF::fillChannelCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof, float offset, int ipad, TH1F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad)
{

  // we will fill the input for the channel-level calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1. / bc;

  for (auto infotof = calibinfotof->begin(); infotof != calibinfotof->end(); infotof++) {
    double dtime = infotof->getDeltaTimePi() - offset; // removing existing offset
    dtime -= (int(dtime * bc_inv + 5.5) - 5) * bc;     // do truncation far (by 5 units) from zero to avoid truncation of negative numbers

    histo->Fill(dtime);
    if (calibTimePad) {
      calibTimePad->push_back(*infotof);
    }
  }
}
//______________________________________________

void CalibTOF::fillChannelTimeSlewingCalib(float offset, int ipad, TH2F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad)
{

  // we will fill the input for the channel-time-slewing calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1. / bc;

  for (auto infotof = calibTimePad->begin(); infotof != calibTimePad->end(); infotof++) {
    double dtime = infotof->getDeltaTimePi() - offset; // removing the already calculated offset; this is needed to
                                                       // fill the time slewing histogram in the correct range
    dtime -= (int(dtime * bc_inv + 5.5) - 5) * bc;     // do truncation far (by 5 units) from zero to avoid truncation of negative numbers

    histo->Fill(TMath::Min(double(infotof->getTot()), 249.9), dtime);
    mHistoChTimeSlewingAll->Fill(infotof->getTot(), dtime);
  }
}
//______________________________________________

float CalibTOF::doChannelCalibration(int ipad, TH1F* histo, TF1* funcChOffset)
{
  // calibrate single channel from histos - offsets

  float integral = histo->Integral();
  if (!integral) {
    return -1; // we skip directly the channels that were switched off online, the PHOS holes...
  }

  int resfit = FitPeak(funcChOffset, histo, 500., 3., 2., "ChannelOffset");

  // return a number greater than zero to distinguish bad fit from empty channels(fraction=0)
  if (resfit) {
    return 0.0001; // fit was not good
  }

  float mean = funcChOffset->GetParameter(1);
  float sigma = funcChOffset->GetParameter(2);
  float intmin = mean - 5 * sigma;
  float intmax = mean + 5 * sigma;
  float intmin2;
  float intmax2;

  // if peak is at the border of our bunch-crossing window (-12.5:12.5 ns)
  // continue to extrapolate gaussian integral from the other border
  float addduetoperiodicity = 0;
  if (intmin < -12500) { // at left border
    intmin2 = intmin + 25000;
    intmin = -12500;
    intmax2 = 12500;
    if (intmin2 > intmax) {
      int binmin2 = histo->FindBin(intmin2);
      int binmax2 = histo->FindBin(intmax2);
      addduetoperiodicity = histo->Integral(binmin2, binmax2);
    }
  } else if (intmax > 12500) { // at right border
    intmax2 = intmax - 25000;
    intmax = 12500;
    intmin2 = -12500;
    if (intmax2 < intmin) {
      int binmin2 = histo->FindBin(intmin2);
      int binmax2 = histo->FindBin(intmax2);
      addduetoperiodicity = histo->Integral(binmin2, binmax2);
    }
  }

  int binmin = histo->FindBin(intmin);
  int binmax = histo->FindBin(intmax);

  if (binmin < 1) {
    binmin = 1; // avoid to take the underflow bin (can happen in case the sigma is too large)
  }
  if (binmax > histo->GetNbinsX()) {
    binmax = histo->GetNbinsX(); // avoid to take the overflow bin (can happen in case the sigma is too large)
  }

  return (histo->Integral(binmin, binmax) + addduetoperiodicity) / integral;
}
//______________________________________________

void CalibTOF::resetChannelLevelHistos(TH1F* histoOffset[NPADSPERSTEP], TH2F* histoTimeSlewing, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad[NPADSPERSTEP])
{

  // reset single channel histos

  for (int ipad = 0; ipad < NPADSPERSTEP; ipad++) {
    histoOffset[ipad]->Reset();
    if (calibTimePad[ipad]) {
      calibTimePad[ipad]->clear();
    }
  }
  histoTimeSlewing->Reset();
}

//______________________________________________

TGraphErrors* CalibTOF::processSlewing(TH2F* histo, Bool_t forceZero, TF1* fitFunc)
{
  /* projection-x */
  TH1D* hpx = histo->ProjectionX("hpx");

  /* define mix and max TOT bin */
  Int_t minBin = hpx->FindFirstBinAbove(0);
  Int_t maxBin = hpx->FindLastBinAbove(0);
  Float_t minTOT = hpx->GetBinLowEdge(minBin);
  Float_t maxTOT = hpx->GetBinLowEdge(maxBin + 1);
  //  printf("min/max TOT defined: %f < TOT < %f ns [%d, %d]\n", minTOT, maxTOT, minBin, maxBin);

  /* loop over TOT bins */
  Int_t nPoints = 0;
  Float_t tot[10000], toterr[10000];
  Float_t mean[10000], meanerr[10000];
  Float_t sigma[10000], vertexSigmaerr[10000];
  for (Int_t ibin = minBin; ibin <= maxBin; ibin++) {

    /* define TOT window */
    Int_t startBin = ibin;
    Int_t endBin = ibin;
    while (hpx->Integral(startBin, endBin) < 300) {
      if (startBin == 1 && forceZero) {
        break;
      }
      if (endBin < maxBin) {
        endBin++;
      } else if (startBin > minBin) {
        startBin--;
      } else {
        break;
      }
    }
    if (hpx->Integral(startBin, endBin) <= 0) {
      continue;
    }
    //    printf("TOT window defined: %f < TOT < %f ns [%d, %d], %d tracks\n", hpx->GetBinLowEdge(startBin), hpx->GetBinLowEdge(endBin + 1), startBin, endBin, (Int_t)hpx->Integral(startBin, endBin));

    /* projection-y */
    TH1D* hpy = histo->ProjectionY("hpy", startBin, endBin);

    /* average TOT */
    hpx->GetXaxis()->SetRange(startBin, endBin);
    tot[nPoints] = hpx->GetMean();
    toterr[nPoints] = hpx->GetMeanError();

    /* fit peak in slices of tot */
    if (FitPeak(fitFunc, hpy, 500., 3., 2., Form("TotBins%04d_%04d", startBin, endBin), histo) != 0) {
      //      printf("troubles fitting time-zero TRACKS, skip\n");
      delete hpy;
      continue;
    }
    mean[nPoints] = fitFunc->GetParameter(1);
    meanerr[nPoints] = fitFunc->GetParError(1);

    /* delete projection-y */
    delete hpy;

    //    printf("meanerr = %f\n",meanerr[nPoints]);

    /* increment n points if good mean error */
    if (meanerr[nPoints] < 100.) {
      nPoints++;
    }

    /* set current bin */
    ibin = endBin;

  } /* end of loop over time bins */

  /* check points */
  if (nPoints <= 0) {
    //    printf("no measurement available, quit\n");
    delete hpx;
    return nullptr;
  }

  /* create graph */
  TGraphErrors* gSlewing = new TGraphErrors(nPoints, tot, mean, toterr, meanerr);

  delete hpx;
  return gSlewing;
}
//______________________________________________

Int_t CalibTOF::FitPeak(TF1* fitFunc, TH1* h, Float_t startSigma, Float_t nSigmaMin, Float_t nSigmaMax, const char* debuginfo, TH2* hdbg)
{
  /*
   * fit peak
   */

  Double_t fitCent = h->GetBinCenter(h->GetMaximumBin());
  if (fitCent < -12500) {
    printf("fitCent = %f (%s). This is wrong, please check!\n", fitCent, h->GetName());
    fitCent = -12500;
  }
  if (fitCent > 12500) {
    printf("fitCent = %f (%s). This is wrong, please check!\n", fitCent, h->GetName());
    fitCent = 12500;
  }
  Double_t fitMin = fitCent - nSigmaMin * startSigma;
  Double_t fitMax = fitCent + nSigmaMax * startSigma;
  if (fitMin < -12500) {
    fitMin = -12500;
  }
  if (fitMax > 12500) {
    fitMax = 12500;
  }
  fitFunc->SetParLimits(1, fitMin, fitMax);
  fitFunc->SetParameter(0, 100);
  fitFunc->SetParameter(1, fitCent);
  fitFunc->SetParameter(2, startSigma);
  Int_t fitres = h->Fit(fitFunc, "WWq0", "", fitMin, fitMax);
  //printf("%s) init: %f %f\n ",h->GetName(),fitMin,fitMax);
  if (fitres != 0) {
    return fitres;
  }
  /* refit with better range */
  for (Int_t i = 0; i < 3; i++) {
    fitCent = fitFunc->GetParameter(1);
    fitMin = fitCent - nSigmaMin * abs(fitFunc->GetParameter(2));
    fitMax = fitCent + nSigmaMax * abs(fitFunc->GetParameter(2));
    if (fitMin < -12500) {
      fitMin = -12500;
    }
    if (fitMax > 12500) {
      fitMax = 12500;
    }
    if (fitMin >= fitMax) {
      printf("%s) step%i: %f %f\n ", h->GetName(), i, fitMin, fitMax);
    }
    fitFunc->SetParLimits(1, fitMin, fitMax);
    fitres = h->Fit(fitFunc, "q0", "", fitMin, fitMax);
    if (fitres != 0) {
      printf("%s) step%i: %f in %f - %f\n ", h->GetName(), i, fitCent, fitMin, fitMax);

      if (mDebugMode > 1) {
        char* filename = Form("TOFDBG_%s.root", h->GetName());
        if (hdbg) {
          filename = Form("TOFDBG_%s_%s.root", hdbg->GetName(), debuginfo);
        }
        //    printf("write %s\n", filename);
        TFile ff(filename, "RECREATE");
        h->Write();
        if (hdbg) {
          hdbg->Write();
        }
        ff.Close();
      }

      return fitres;
    }
  }

  if (mDebugMode > 1 && fitFunc->GetParError(1) > 100) {
    char* filename = Form("TOFDBG_%s.root", h->GetName());
    if (hdbg) {
      filename = Form("TOFDBG_%s_%s.root", hdbg->GetName(), debuginfo);
    }
    //    printf("write %s\n", filename);
    TFile ff(filename, "RECREATE");
    h->Write();
    if (hdbg) {
      hdbg->Write();
    }
    ff.Close();
  }

  return fitres;
}
//______________________________________________

void CalibTOF::merge(const char* name)
{
  TFile* f = TFile::Open(name);
  if (!f) {
    LOG(error) << "File " << name << "not found (merging skept)";
    return;
  }
  TTree* t = (TTree*)f->Get(mOutputTree->GetName());
  if (!t) {
    LOG(error) << "Tree " << mOutputTree->GetName() << "not found in " << name << " (merging skept)";
    return;
  }
  t->ls();
  mOutputTree->ls();

  o2::dataformats::CalibLHCphaseTOF* LHCphaseObj = nullptr;

  o2::dataformats::CalibTimeSlewingParamTOF* timeSlewingObj = nullptr;

  t->SetBranchAddress("mLHCphaseObj", &LHCphaseObj);
  t->SetBranchAddress("mTimeSlewingObj", &timeSlewingObj);

  t->GetEvent(0);

  *mTimeSlewingObj += *timeSlewingObj;
  *mLHCphaseObj += *LHCphaseObj;
  f->Close();
}
//______________________________________________

void CalibTOF::flagProblematics()
{

  // method to flag problematic channels: Fraction, Sigma -> all negative if channel is bad (otherwise all positive)

  TH1F* hsigmapeak = new TH1F("hsigmapeak", ";#sigma_{peak} (ps)", 1000, 0, 1000);
  TH1F* hfractionpeak = new TH1F("hfractionpeak", ";fraction under peak", 1001, 0, 1.01);

  int ipad;
  float sigmaMin, sigmaMax, fractionMin;

  int nActiveChannels = 0;
  int nGoodChannels = 0;

  TF1* fFuncSigma = new TF1("fFuncSigma", "TMath::Gaus(x,[1],[2])*[0]*(x<[1]) + TMath::Gaus(x,[1],[3])*[0]*(x>[1])");
  fFuncSigma->SetParameter(0, 1000);
  fFuncSigma->SetParameter(1, 200);
  fFuncSigma->SetParameter(2, 200);
  fFuncSigma->SetParameter(3, 200);

  TF1* fFuncFraction = new TF1("fFuncFraction", "TMath::Gaus(x,[1],[2])*[0]*(x<[1]) + TMath::Gaus(x,[1],[3])*[0]*(x>[1])");
  fFuncFraction->SetParameter(0, 1000);
  fFuncFraction->SetParameter(1, 0.8);
  fFuncFraction->SetParameter(2, 0.1);
  fFuncFraction->SetParameter(3, 0.1);

  // we group pads according to the z-coordinate since we noted a z-dependence in sigmas and fraction-under-peak
  for (int iz = 0; iz < o2::tof::Geo::NSTRIPXSECTOR * 2; iz++) {
    hsigmapeak->Reset();
    hfractionpeak->Reset();

    for (int k = 0; k < o2::tof::Geo::NPADX; k++) {
      ipad = 48 * iz + k;

      for (int i = 0; i < 18; i++) {
        // exclude channel without entries
        if (mTimeSlewingObj->getFractionUnderPeak(i, ipad) < 0) {
          continue;
        }

        nActiveChannels++;

        hsigmapeak->Fill(mTimeSlewingObj->getSigmaPeak(i, ipad));
        hfractionpeak->Fill(mTimeSlewingObj->getFractionUnderPeak(i, ipad));
      }
    }

    hsigmapeak->Fit(fFuncSigma, "WWq0");
    hfractionpeak->Fit(fFuncFraction, "WWq0");

    sigmaMin = fFuncSigma->GetParameter(1) - mNsigmaSigmaProblematicCut * abs(fFuncSigma->GetParameter(2));
    sigmaMax = fFuncSigma->GetParameter(1) + mNsigmaSigmaProblematicCut * abs(fFuncSigma->GetParameter(3));
    fractionMin = fFuncFraction->GetParameter(1) - mNsigmaFractionProblematicCut * abs(fFuncFraction->GetParameter(2));

    for (int k = 0; k < o2::tof::Geo::NPADX; k++) {
      ipad = 48 * iz + k;

      for (int i = 0; i < 18; i++) {
        // exclude channel without entries
        if (mTimeSlewingObj->getFractionUnderPeak(i, ipad) < 0) {
          continue;
        }

        if (mTimeSlewingObj->getSigmaPeak(i, ipad) < sigmaMin ||
            mTimeSlewingObj->getSigmaPeak(i, ipad) > sigmaMax ||
            mTimeSlewingObj->getFractionUnderPeak(i, ipad) < fractionMin) {
          mTimeSlewingObj->setFractionUnderPeak(i, ipad, -mTimeSlewingObj->getFractionUnderPeak(i, ipad));
          mTimeSlewingObj->setSigmaPeak(i, ipad, -mTimeSlewingObj->getSigmaPeak(i, ipad));
        } else {
          nGoodChannels++;
        }
      }
    }
  }

  Printf("Check for TOF problematics: nActiveChannels=%d - nGoodChannels=%d - fractionGood = %f", int(nActiveChannels), int(nGoodChannels), nGoodChannels * 1. / nActiveChannels);
}
