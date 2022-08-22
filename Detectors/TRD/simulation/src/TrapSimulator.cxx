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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD MCM (Multi Chip Module) simulator                                    //
//  which simulates the TRAP processing after the AD-conversion.             //
//  The relevant parameters (i.e. configuration settings of the TRAP)        //
//  are taken from TrapConfig.                                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDSimulation/SimParam.h"
#include "TRDBase/CalOnlineGainTables.h"
#include "TRDSimulation/TrapConfigHandler.h"
#include "TRDSimulation/TrapSimulator.h"
#include "fairlogger/Logger.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"

#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLine.h"
#include "TMath.h"
#include "TRandom.h"
#include "TFile.h"

#include <iomanip>

using namespace o2::trd;
using namespace std;
using namespace o2::trd::constants;

const int TrapSimulator::mgkFormatIndex = std::ios_base::xalloc();
const std::array<unsigned short, 4> TrapSimulator::mgkFPshifts{11, 14, 17, 21};

void TrapSimulator::init(TrapConfig* trapconfig, int det, int robPos, int mcmPos)
{
  //
  // Initialize the class with new MCM position information
  //
  mDetector = det;
  mRobPos = robPos;
  mMcmPos = mcmPos;

  uint64_t row = mFeeParam->getPadRowFromMCM(mRobPos, mMcmPos); // need uint64_t type to assemble mTrkltWordEmpty below
  uint64_t column = mMcmPos % NMCMROBINCOL;
  // prepare a part of the MCM header, what still is missing are the 3 x 8 bits from the charges
  // < 1 | padrow (4 bits) | column (2 bits) | 00..0 (8 bit) | 00..0 (8 bit) | 00..0 (8 bit) | 1 >
  mMcmHeaderEmpty = (1 << 31) | (row << 27) | (column << 25) | 1;
  // prepare the part of the Tracklet64 which is common to all tracklets of this MCM
  uint64_t hcid = 2 * mDetector + (mRobPos % 2);
  uint64_t format = mUseFloatingPointForQ ? 1UL : 0UL;
  mTrkltWordEmpty = (format << Tracklet64::formatbs) | (hcid << Tracklet64::hcidbs) | (row << Tracklet64::padrowbs) | (column << Tracklet64::colbs);

  if (!mInitialized) {
    mTrapConfig = trapconfig;
    mNTimeBin = mTrapConfig->getTrapReg(TrapConfig::kC13CPUA, mDetector, mRobPos, mMcmPos);
    mZSMap.resize(NADCMCM);
    mADCR.resize(mNTimeBin * NADCMCM);
    mADCF.resize(mNTimeBin * NADCMCM);
  }

  mInitialized = true;
  reset();
}

void TrapSimulator::reset()
{
  // Resets the data values and internal filter registers
  // by re-initialising them

  if (!checkInitialized()) {
    return;
  }

  //clear the adc data
  std::fill(mADCR.begin(), mADCR.end(), 0);
  std::fill(mADCF.begin(), mADCF.end(), 0);
  std::fill(mADCDigitIndices.begin(), mADCDigitIndices.end(), -1);

  for (auto filterreg : mInternalFilterRegisters) {
    filterreg.ClearReg();
  }

  // Default unread, low active bit mask
  std::fill(mZSMap.begin(), mZSMap.end(), 0);
  std::fill(mMCMT.begin(), mMCMT.end(), 0);

  filterPedestalInit();
  //filterGainInit(); // we do not use the gain filter anyway, so disable it completely
  filterTailInit();

  for (auto& fitreg : mFitReg) {
    fitreg.ClearReg();
  }
  mADCFilled = 0;

  mTrackletArray64.clear();
  mTrackletDigitCount.clear();
  mTrackletDigitIndices.clear();

  mDataIsSet = false;
}

// ----- I/O implementation -----

ostream& TrapSimulator::text(ostream& os)
{
  // manipulator to activate output in text format (default)

  os.iword(mgkFormatIndex) = 0;
  return os;
}

ostream& TrapSimulator::cfdat(ostream& os)
{
  // manipulator to activate output in CFDAT format
  // to send to the FEE via SCSN

  os.iword(mgkFormatIndex) = 1;
  return os;
}

ostream& TrapSimulator::raw(ostream& os)
{
  // manipulator to activate output as raw data dump

  os.iword(mgkFormatIndex) = 2;
  return os;
}

//std::ostream& operator<<(std::ostream& os, const TrapSimulator& mcm); // data output using ostream (e.g. cout << mcm;)
std::ostream& o2::trd::operator<<(std::ostream& os, const TrapSimulator& mcm)
{
  // output implementation

  // no output for non-initialized MCM
  if (!mcm.checkInitialized()) {
    return os;
  }

  // ----- human-readable output -----
  if (os.iword(TrapSimulator::mgkFormatIndex) == 0) {

    os << "TRAP " << mcm.getMcmPos() << " on ROB " << mcm.getRobPos() << " in detector " << mcm.getDetector() << std::endl;

    os << "----- Unfiltered ADC data (10 bit) -----" << std::endl;
    os << "ch    ";
    for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
      os << std::setw(5) << iChannel;
    }
    os << std::endl;
    for (int iTimeBin = 0; iTimeBin < mcm.getNumberOfTimeBins(); iTimeBin++) {
      os << "tb " << std::setw(2) << iTimeBin << ":";
      for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
        os << std::setw(5) << (mcm.getDataRaw(iChannel, iTimeBin) >> mcm.mgkAddDigits);
      }
      os << std::endl;
    }
    os << "----- Filtered ADC data (10+2 bit) -----" << std::endl;
    os << "ch    ";
    for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
      os << std::setw(4) << iChannel
         << ((~mcm.getZeroSupressionMap(iChannel) != 0) ? "!" : " ");
    }
    os << std::endl;
    for (int iTimeBin = 0; iTimeBin < mcm.getNumberOfTimeBins(); iTimeBin++) {
      os << "tb " << std::setw(2) << iTimeBin << ":";
      for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
        os << std::setw(4) << (mcm.getDataFiltered(iChannel, iTimeBin))
           << (((mcm.getZeroSupressionMap(iChannel) & (1 << iTimeBin)) == 0) ? "!" : " ");
      }
      os << std::endl;
    }
  }

  // ----- CFDAT output -----
  else if (os.iword(TrapSimulator::mgkFormatIndex) == 1) {
    int dest = 127;
    int addrOffset = 0x2000;
    int addrStep = 0x80;

    for (int iTimeBin = 0; iTimeBin < mcm.getNumberOfTimeBins(); iTimeBin++) {
      for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
        os << std::setw(5) << 10
           << std::setw(5) << addrOffset + iChannel * addrStep + iTimeBin
           << std::setw(5) << (mcm.getDataFiltered(iChannel, iTimeBin))
           << std::setw(5) << dest << std::endl;
      }
      os << std::endl;
    }
  }

  // ----- raw data ouptut -----
  else if (os.iword(TrapSimulator::mgkFormatIndex) == 2) {
    int bufSize = 300;
    std::vector<uint32_t> buf;
    buf.reserve(bufSize);
    int bufLength = mcm.packData(buf, 0);

    for (int i = 0; i < bufLength; i++) {
      std::cout << "0x" << std::hex << buf[i] << std::dec << std::endl;
    }

  }

  else {
    os << "unknown format set" << std::endl;
  }

  return os;
}

void TrapSimulator::printFitRegXml(ostream& os) const
{
  // print fit registres in XML format

  bool tracklet = false;

  for (int cpu = 0; cpu < 4; cpu++) {
    if (mFitPtr[cpu] != NOTRACKLETFIT) {
      tracklet = true;
    }
  }
  const char* const aquote{R"(")"};
  const char* const cmdid{R"(" cmdid="0">)"};
  if (tracklet == true) {
    os << "<nginject>" << std::endl;
    os << "<ack roc=\"" << mDetector << cmdid << std::endl;
    os << "<dmem-readout>" << std::endl;
    os << "<d det=\"" << mDetector << "\">" << std::endl;
    os << " <ro-board rob=\"" << mRobPos << "\">" << std::endl;
    os << "  <m mcm=\"" << mMcmPos << "\">" << std::endl;

    for (int cpu = 0; cpu < 4; cpu++) {
      os << "   <c cpu=\"" << cpu << "\">" << std::endl;
      if (mFitPtr[cpu] != NOTRACKLETFIT) {
        for (int adcch = mFitPtr[cpu]; adcch < mFitPtr[cpu] + 2; adcch++) {
          if (adcch > 24) {
            LOG(error) << "adcch going awol : " << adcch << " > 25";
          }
          os << "    <ch chnr=\"" << adcch << "\">" << std::endl;
          os << "     <hits>" << mFitReg[adcch].nHits << "</hits>" << std::endl;
          os << "     <q0>" << mFitReg[adcch].q0 << "</q0>" << std::endl;
          os << "     <q1>" << mFitReg[adcch].q1 << "</q1>" << std::endl;
          os << "     <sumx>" << mFitReg[adcch].sumX << "</sumx>" << std::endl;
          os << "     <sumxsq>" << mFitReg[adcch].sumX2 << "</sumxsq>" << std::endl;
          os << "     <sumy>" << mFitReg[adcch].sumY << "</sumy>" << std::endl;
          os << "     <sumysq>" << mFitReg[adcch].sumY2 << "</sumysq>" << std::endl;
          os << "     <sumxy>" << mFitReg[adcch].sumXY << "</sumxy>" << std::endl;
          os << "    </ch>" << std::endl;
        }
      }
      os << "      </c>" << std::endl;
    }
    os << "    </m>" << std::endl;
    os << "  </ro-board>" << std::endl;
    os << "</d>" << std::endl;
    os << "</dmem-readout>" << std::endl;
    os << "</ack>" << std::endl;
    os << "</nginject>" << std::endl;
  }
}

void TrapSimulator::printTrackletsXml(ostream& os) const
{
  // print tracklets in XML format
  //TODO FIX this to new format
  const char* const cmdid{R"(" cmdid="0">)"};
  os << "<nginject>" << std::endl;
  os << "<ack roc=\"" << mDetector << cmdid << std::endl;
  os << "<dmem-readout>" << std::endl;
  os << "<d det=\"" << mDetector << "\">" << std::endl;
  os << "  <ro-board rob=\"" << mRobPos << "\">" << std::endl;
  os << "    <m mcm=\"" << mMcmPos << "\">" << std::endl;

  int pid, padrow, slope, offset;
  //TODO take care of the 0th being the header, and cpu1,2,3 being 1 to 3 tracklets.
  for (int cpu = 0; cpu < 4; cpu++) {
    if (mMCMT[cpu] == 0x10001000) {
      pid = -1;
      padrow = -1;
      slope = -1;
      offset = -1;
    } else {
      pid = (mMCMT[cpu] & 0xFF000000) >> 24;
      padrow = (mMCMT[cpu] & 0xF00000) >> 20;
      slope = (mMCMT[cpu] & 0xFE000) >> 13;
      offset = (mMCMT[cpu] & 0x1FFF);
    }
    os << "      <trk> <pid>" << pid << "</pid>"
       << " <padrow>" << padrow << "</padrow>"
       << " <slope>" << slope << "</slope>"
       << " <offset>" << offset << "</offset>"
       << "</trk>" << std::endl;
  }

  os << "    </m>" << std::endl;
  os << "  </ro-board>" << std::endl;
  os << "</d>" << std::endl;
  os << "</dmem-readout>" << std::endl;
  os << "</ack>" << std::endl;
  os << "</nginject>" << std::endl;
}

void TrapSimulator::printAdcDatTxt(ostream& os) const
{
  // print ADC data in text format (suitable as Modelsim stimuli)

  os << "# MCM " << mMcmPos << " on ROB " << mRobPos << " in detector " << mDetector << std::endl;

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iChannel = 0; iChannel < NADCMCM; ++iChannel) {
      os << std::setw(5) << (getDataRaw(iChannel, iTimeBin) >> mgkAddDigits);
    }
    os << std::endl;
  }
}

void TrapSimulator::printAdcDatHuman(ostream& os) const
{
  // print ADC data in human-readable format

  os << "MCM " << mMcmPos << " on ROB " << mRobPos << " in detector " << mDetector << std::endl;

  os << "----- Unfiltered ADC data (10 bit) -----" << std::endl;
  os << "ch    ";
  for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
    os << std::setw(5) << iChannel;
  }
  os << std::endl;
  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    os << "tb " << std::setw(2) << iTimeBin << ":";
    for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
      os << std::setw(5) << (getDataRaw(iChannel, iTimeBin));
    }
    os << std::endl;
  }

  os << "----- Filtered ADC data (10+2 bit) -----" << std::endl;
  os << "ch    ";
  for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
    os << std::setw(4) << iChannel
       << ((~mZSMap[iChannel] != 0) ? "!" : " ");
  }
  os << std::endl;
  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    os << "tb " << std::setw(2) << iTimeBin << ":";
    for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
      os << std::setw(4) << (getDataFiltered(iChannel, iTimeBin))
         << (((mZSMap[iChannel] & (1 << iTimeBin)) == 0) ? "!" : " ");
    }
    os << std::endl;
  }
}

void TrapSimulator::printAdcDatXml(ostream& os) const
{
  // print ADC data in XML format

  const char* const cmdid{R"(" cmdid="0">)"};
  os << "<nginject>" << std::endl;
  os << "<ack roc=\"" << mDetector << cmdid << std::endl;
  os << "<dmem-readout>" << std::endl;
  os << "<d det=\"" << mDetector << "\">" << std::endl;
  os << " <ro-board rob=\"" << mRobPos << "\">" << std::endl;
  os << "  <m mcm=\"" << mMcmPos << "\">" << std::endl;

  for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
    os << "   <ch chnr=\"" << iChannel << "\">" << std::endl;
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      os << "<tb>" << mADCF[iChannel * mNTimeBin + iTimeBin] / 4 << "</tb>";
    }
    os << "   </ch>" << std::endl;
  }

  os << "  </m>" << std::endl;
  os << " </ro-board>" << std::endl;
  os << "</d>" << std::endl;
  os << "</dmem-readout>" << std::endl;
  os << "</ack>" << std::endl;
  os << "</nginject>" << std::endl;
}

void TrapSimulator::printAdcDatDatx(ostream& os, bool broadcast, int timeBinOffset) const
{
  // print ADC data in datx format (to send to FEE)

  mTrapConfig->printDatx(os, 2602, 1, 0, 127); // command to enable the ADC clock - necessary to write ADC values to MCM
  os << std::endl;

  int addrOffset = 0x2000;
  int addrStep = 0x80;
  int addrOffsetEBSIA = 0x20;

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iChannel = 0; iChannel < NADCMCM; iChannel++) {
      if ((iTimeBin < timeBinOffset) || (iTimeBin >= mNTimeBin + timeBinOffset)) {
        if (broadcast == false) {
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, 10, getRobPos(), getMcmPos());
        } else {
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, 10, 0, 127);
        }
      } else {
        if (broadcast == false) {
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, (getDataFiltered(iChannel, iTimeBin - timeBinOffset) / 4), getRobPos(), getMcmPos());
        } else {
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, (getDataFiltered(iChannel, iTimeBin - timeBinOffset) / 4), 0, 127);
        }
      }
    }
    os << std::endl;
  }
}

void TrapSimulator::noiseTest(int nsamples, int mean, int sigma, int inputGain, int inputTail)
{
  // This function can be used to test the filters.
  // It feeds nsamples of ADC values with a gaussian distribution specified by mean and sigma.
  // The filter chain implemented here consists of:
  // Pedestal -> Gain -> Tail
  // With inputGain and inputTail the input to the gain and tail filter, respectively,
  // can be chosen where
  // 0: noise input
  // 1: pedestal output
  // 2: gain output
  // The input has to be chosen from a stage before.
  // The filter behaviour is controlled by the TRAP parameters from TrapConfig in the
  // same way as in normal simulation.
  // The functions produces four histograms with the values at the different stages.

  if (!checkInitialized()) {
    return;
  }

  std::string nameInputGain;
  std::string nameInputTail;

  switch (inputGain) {
    case 0:
      nameInputGain = "Noise";
      break;

    case 1:
      nameInputGain = "Pedestal";
      break;

    default:
      LOG(error) << "Undefined input to tail cancellation filter";
      return;
  }

  switch (inputTail) {
    case 0:
      nameInputTail = "Noise";
      break;

    case 1:
      nameInputTail = "Pedestal";
      break;

    case 2:
      nameInputTail = "Gain";
      break;

    default:
      LOG(error) << "Undefined input to tail cancellation filter";
      return;
  }

  TH1F* h = new TH1F("noise", "Gaussian Noise;sample;ADC count",
                     nsamples, 0, nsamples);
  TH1F* hfp = new TH1F("ped", "Noise #rightarrow Pedestal filter;sample;ADC count", nsamples, 0, nsamples);
  TH1F* hfg = new TH1F("gain",
                       (nameInputGain + "#rightarrow Gain;sample;ADC count").c_str(),
                       nsamples, 0, nsamples);
  TH1F* hft = new TH1F("tail",
                       (nameInputTail + "#rightarrow Tail;sample;ADC count").c_str(),
                       nsamples, 0, nsamples);
  h->SetStats(false);
  hfp->SetStats(false);
  hfg->SetStats(false);
  hft->SetStats(false);

  for (int i = 0; i < nsamples; i++) {
    int value;                               // ADC count with noise (10 bit)
    int valuep;                              // pedestal filter output (12 bit)
    int valueg;                              // gain filter output (12 bit)
    int valuet;                              // tail filter value (12 bit)
    value = (int)gRandom->Gaus(mean, sigma); // generate noise with gaussian distribution
    h->SetBinContent(i, value);

    valuep = filterPedestalNextSample(1, 0, ((int)value) << 2);

    if (inputGain == 0) {
      valueg = filterGainNextSample(1, ((int)value) << 2);
    } else {
      valueg = filterGainNextSample(1, valuep);
    }

    if (inputTail == 0) {
      valuet = filterTailNextSample(1, ((int)value) << 2);
    } else if (inputTail == 1) {
      valuet = filterTailNextSample(1, valuep);
    } else {
      valuet = filterTailNextSample(1, valueg);
    }

    hfp->SetBinContent(i, valuep >> 2);
    hfg->SetBinContent(i, valueg >> 2);
    hft->SetBinContent(i, valuet >> 2);
  }

  TCanvas* c = new TCanvas;
  c->Divide(2, 2);
  c->cd(1);
  h->Draw();
  c->cd(2);
  hfp->Draw();
  c->cd(3);
  hfg->Draw();
  c->cd(4);
  hft->Draw();
}


void TrapSimulator::print(int choice) const
{
  // Prints the data stored and/or calculated for this TRAP
  // The output is controlled by option which can be any bitpattern defined in the header
  // PRINTRAW - prints raw ADC data
  // PRINTFILTERED - prints filtered data
  // PRINTHITS - prints detected hits
  // PRINTTRACKLETS - prints found tracklets
  // The later stages are only meaningful after the corresponding calculations
  // have been performed.
  // Codacy wont let us use string choices, as it says we must use string::starts_with() instead of string::find()

  if (!checkInitialized()) {
    return;
  }

  LOG(debug) << "MCM " << mMcmPos << "  on ROB " << mRobPos << " in detector " << mDetector;

  //std::string opt = option;
  if ((choice & PRINTRAW) != 0 || (choice & PRINTFILTERED) != 0) {
    std::cout << *this;
  }

  if ((choice & PRINTFOUND) != 0) {
    LOG(debug) << "Found Tracklets:";
    for (int iTrkl = 0; iTrkl < mTrackletArray64.size(); iTrkl++) {
      LOG(debug) << "tracklet " << iTrkl << ": 0x" << hex << std::setw(32) << mTrackletArray64[iTrkl].getTrackletWord();
      LOG(debug) << mTrackletArray64[iTrkl];
    }
  }
}

void TrapSimulator::draw(int choice, int index)
{
  // Plots the data stored in a 2-dim. timebin vs. ADC channel plot.
  // The choice selects what data is plotted and is enumated in the header.
  // PLOTRAW - plot raw data (default)
  // PLOTFILTERED - plot filtered data (meaningless if R is specified)
  // In addition to the ADC values:
  // PLOTHITS - plot hits
  // PLOTTRACKLETS - plot tracklets
  if (!checkInitialized()) {
    return;
  }
  TFile* rootfile = new TFile("trdtrackletplots.root", "UPDATE");
  TCanvas* c1 = new TCanvas(Form("canvas_%i_%i:%i:%i_%i", index, mDetector, mRobPos, mMcmPos, (int)mTrackletArray64.size()));
  TH2F* hist = new TH2F(Form("mcmdata_%i", index),
                        Form("Data of MCM %i on ROB %i in detector %i ", mMcmPos, mRobPos, mDetector),
                        NADCMCM,
                        -0.5,
                        NADCMCM - 0.5,
                        getNumberOfTimeBins(),
                        -0.5,
                        getNumberOfTimeBins() - 0.5);
  hist->GetXaxis()->SetTitle("ADC Channel");
  hist->GetYaxis()->SetTitle("Timebin");
  hist->SetStats(false);
  TH2F* histfiltered = new TH2F(Form("mcmdataf_%i", index),
                                Form("Data of MCM %i on ROB %i in detector %i filtered", mMcmPos, mRobPos, mDetector),
                                NADCMCM,
                                -0.5,
                                NADCMCM - 0.5,
                                getNumberOfTimeBins(),
                                -0.5,
                                getNumberOfTimeBins() - 0.5);
  histfiltered->GetXaxis()->SetTitle("ADC Channel");
  histfiltered->GetYaxis()->SetTitle("Timebin");

  if ((choice & PLOTRAW) != 0) {
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      for (int iAdc = 0; iAdc < NADCMCM; iAdc++) {
        hist->SetBinContent(iAdc + 1, iTimeBin + 1, mADCR[iAdc * mNTimeBin + iTimeBin] >> mgkAddDigits);
      }
    }
    hist->Draw("COLZ");
  } else {
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      for (int iAdc = 0; iAdc < NADCMCM; iAdc++) {
        histfiltered->SetBinContent(iAdc + 1, iTimeBin + 1, mADCF[iAdc * mNTimeBin + iTimeBin] >> mgkAddDigits);
      }
    }
    histfiltered->Draw("COLZ");
  }

  if ((choice & PLOTTRACKLETS) != 0) {
    TLine* trklLines = new TLine[4];
    LOG(debug) << "Tracklet start for index : " << index;
    if (mTrackletArray64.size() > 0) {
      LOG(debug) << "Tracklet : for " << mTrackletArray64[0].getDetector() << "::" << mTrackletArray64[0].getROB() << " : " << mTrackletArray64[0].getMCM();
    } else {
      LOG(debug) << "Tracklet : for trackletarray size of zero ";
    }
    for (int iTrkl = 0; iTrkl < mTrackletArray64.size(); iTrkl++) {
      Tracklet64 trkl = mTrackletArray64[iTrkl];
      float position = trkl.getPosition();
      int ndrift = mTrapConfig->getDmemUnsigned(mgkDmemAddrNdrift, mDetector, mRobPos, mMcmPos) >> 5;
      float slope = trkl.getSlope();

      int t0 = mTrapConfig->getTrapReg(TrapConfig::kTPFS, mDetector, mRobPos, mMcmPos);
      int t1 = mTrapConfig->getTrapReg(TrapConfig::kTPFE, mDetector, mRobPos, mMcmPos);

      trklLines[iTrkl].SetX1(position - slope * t0);
      trklLines[iTrkl].SetY1(t0);
      trklLines[iTrkl].SetX2(position - slope * t1);
      trklLines[iTrkl].SetY2(t1);
      trklLines[iTrkl].SetLineColor(2);
      trklLines[iTrkl].SetLineWidth(2);
      LOG(debug) << "Tracklet " << iTrkl << ": y = " << trkl.getPosition() << ", slope = " << (float)trkl.getSlope() << "for a det:rob:mcm combo of : " << mDetector << ":" << mRobPos << ":" << mMcmPos;
      LOG(debug) << "Tracklet " << iTrkl << ": x1,y1,x2,y2 :: " << trklLines[iTrkl].GetX1() << "," << trklLines[iTrkl].GetY1() << "," << trklLines[iTrkl].GetX2() << "," << trklLines[iTrkl].GetY2();
      LOG(debug) << "Tracklet " << iTrkl << ": t0 : " << t0 << ", t1 " << t1 << ", slope:" << slope << ",  which comes from : " << mTrapConfig->getDmemUnsigned(mgkDmemAddrNdrift, mDetector, mRobPos, mMcmPos) << " shifted 5 to the right ";
      trklLines[iTrkl].Draw();
    }
    LOG(debug) << "Tracklet end ...";
  }
  c1->Write();
  rootfile->Close();
}

void TrapSimulator::setData(int adc, const ArrayADC& data, unsigned int digitIdx)
{
  //
  // Store ADC data into array of raw data
  //

  if (!checkInitialized()) {
    LOG(error) << "TRAP is not initialized, cannot call setData()";
    return;
  }

  if (adc < 0 || adc >= NADCMCM) {
    LOG(error) << "Error: ADC " << adc << " is out of range (0 .. " << NADCMCM - 1 << ")";
    return;
  }

  for (int it = 0; it < mNTimeBin; it++) {
    mADCR[adc * mNTimeBin + it] = (data[it] << mgkAddDigits) + (mAdditionalBaseline << mgkAddDigits);
    mADCF[adc * mNTimeBin + it] = (data[it] << mgkAddDigits) + (mAdditionalBaseline << mgkAddDigits);
  }
  mDataIsSet = true;
  mADCFilled |= (1 << adc);

  mADCDigitIndices[adc] = digitIdx;
}

void TrapSimulator::setBaselines()
{
  //This function exists as in the old simulator, when data was fed into the trapsim, it was done via the whole adc block for the mcm.
  //if the data was zero or out of spec, the baselines were added
  //we now add it by singular ADC, so there are adc channels that never get touched.
  //this fixes that.

  if (!checkInitialized()) {
    LOG(error) << "TRAP is not initialized, cannot call setBaselines()";
    return;
  }

  //loop over all adcs.
  for (int adc = 0; adc < NADCMCM; adc++) {
    if ((mADCFilled & (1 << adc)) == 0) { // adc is empty by construction of mADCFilled.
      for (int timebin = 0; timebin < mNTimeBin; timebin++) {
        // kFPNP = 32 = 8 << 2 (pedestal correction additive) and kTPFP = 40 = 10 << 2 (filtered pedestal)
        mADCR[adc * mNTimeBin + timebin] = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos); // OS: not using FPNP here, since in filter() the ADC values from the 'raw' array will be copied into the filtered array
        mADCF[adc * mNTimeBin + timebin] = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos);
      }
    }
  }
}

void TrapSimulator::setDataPedestal(int adc)
{
  //
  // Store ADC data into array of raw data
  //

  if (!checkInitialized()) {
    return;
  }

  if (adc < 0 || adc >= NADCMCM) {
    return;
  }

  for (int it = 0; it < mNTimeBin; it++) {
    mADCR[adc * mNTimeBin + it] = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos);
    mADCF[adc * mNTimeBin + it] = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos);
  }
}

//TODO figure why I could not get span to work here
int TrapSimulator::packData(std::vector<uint32_t>& rawdata, uint32_t offset) const
{
  // return # of 32 bit words.
  //
  //given the, up to 3 tracklets, pack them according to the define data format.
  //
  //  std::cout << "span size in packData is : " << rawdata.size() << std::endl;
  //TODO this is left blank so that the dataformats etc. can come in a seperate PR
  //to keep different work seperate.
  uint32_t wordswritten = 0; // count the 32 bit words written;
                             //  std::cout << &raw[offset] << std::endl;
                             //  std::cout << raw.data() << std::endl;;
                             //  //TODO query? the mCPU have the packed data, rather use them?
  TrackletMCMHeader mcmhead;
  int trackletcount = 0;
  // TrackletMCMHeader mcmhead;
  offset++;
  std::array<TrackletMCMData, 3> tracklets{};
  mcmhead.oneb = 1;
  mcmhead.onea = 1;
  mcmhead.padrow = ((mRobPos >> 1) << 2) | (mMcmPos >> 2);
  int mcmcol = mMcmPos % NMCMROBINCOL + (mRobPos % 2) * NMCMROBINCOL;
  int padcol = mcmcol * NCOLMCM + NCOLMCM + 1;
  mcmhead.col = 1; //TODO check this, cant call FeeParam due to virtual function
  LOG(debug) << "packing data with trackletarry64 size of : " << mTrackletArray64.size();
  for (int i = 0; i < 3; i++) {
    if (i < mTrackletArray64.size()) { // we have  a tracklet
      LOG(debug) << "we have a tracklet at i=" << i << " with trackletword 0x" << mTrackletArray64[i].getTrackletWord();
      switch (i) {
        case 0:
          mcmhead.pid0 = ((mTrackletArray64[0].getQ2()) << 2) + ((mTrackletArray64[0].getQ1()) >> 5);
          break; // all of Q2 and upper 2 bits of Q1.
        case 1:
          mcmhead.pid1 = ((mTrackletArray64[1].getQ2()) << 2) + ((mTrackletArray64[1].getQ1()) >> 5);
          break; // all of Q2 and upper 2 bits of Q1.
        case 2:
          mcmhead.pid2 = ((mTrackletArray64[2].getQ2()) << 2) + ((mTrackletArray64[2].getQ1()) >> 5);
          break; // all of Q2 and upper 2 bits of Q1.
      }
      tracklets[i].checkbit = 1;
      LOG(debug) << mTrackletArray64[i];
      tracklets[i].pid = mTrackletArray64[i].getQ0() & (mTrackletArray64[i].getQ1() << 8);
      tracklets[i].slope = mTrackletArray64[i].getSlope();
      tracklets[i].pos = mTrackletArray64[i].getPosition();
      tracklets[i].checkbit = 0;
      trackletcount++;
    } else { // else we dont have a tracklet so mark it off in the header.
      switch (i) {
        case 1:
          mcmhead.pid1 = 0xff;
          LOG(debug) << "setting mcmhead pid1 to 0xff with tracklet array size of " << mTrackletArray64[i];
          break; // set the pid to ff to signify not there
        case 2:
          mcmhead.pid2 = 0xff;
          break; // set the pid to maximal to signify not there (6bits).
      }
    }
  }
  //  raw.push_back((uint32_t)mcmhead)
  LOG(debug) << "pushing back mcm head of 0x" << std::hex << mcmhead.word << " with trackletcount of : " << std::dec << trackletcount << ":-:" << wordswritten;
  rawdata.push_back(mcmhead.word); //memcpy(&rawdata[wordswritten++], &mcmhead, sizeof(mcmhead));
  wordswritten++;
  for (int i = 0; i < trackletcount; i++) {
    LOG(debug) << "pushing back mcmtrackletword of 0x" << std::hex << tracklets[i].word;
    rawdata.push_back(tracklets[i].word); //memcpy(&rawdata[wordswritten++], &tracklets[i], sizeof(TrackletMCMData));
    wordswritten++;
  }

  //display the headers written
  LOG(debug) << ">>>>> START info OUTPUT OF packData trackletcount:-:wordcount" << trackletcount << ":-:" << wordswritten;
  o2::trd::printTrackletMCMHeader(mcmhead);
  o2::trd::printTrackletMCMData(tracklets[0]);
  if (trackletcount > 1) {
    o2::trd::printTrackletMCMData(tracklets[1]);
  }
  if (trackletcount > 2) {
    o2::trd::printTrackletMCMData(tracklets[2]);
  }
  LOG(debug) << "<<<<<  END info OUTPUT OF packData";
  //must produce between 2 and 4 words ... 1 and 3 tracklets.
  //  assert(wordswritten<5);
  //  assert(wordswritten>1);
  LOG(debug) << "now to leave pack data after passing asserts with wordswritten = " << wordswritten;
  return wordswritten; // in units of 32 bits.
}

void TrapSimulator::filter()
{
  //
  // Filter the raw ADC values. The active filter stages and their
  // parameters are taken from TrapConfig.
  // The raw data is stored separate from the filtered data. Thus,
  // it is possible to run the filters on a set of raw values
  // sequentially for parameter tuning.
  //

  // OS: with the current TRAP config and settings both pedestal and tail filter are bypassed

  if (!checkInitialized()) {
    return;
  }

  // Apply filters sequentially. Bypass is handled by filters
  // since counters and internal registers may be updated even
  // if the filter is bypassed.
  // The first filter takes the data from mADCR and
  // outputs to mADCF.

  // Non-linearity filter not implemented.
  filterPedestal();
  //filterGain(); // we do not use the gain filter anyway, so disable it completely
  filterTail();
  // Crosstalk filter not implemented.
}

void TrapSimulator::filterPedestalInit(int baseline)
{
  // Initializes the pedestal filter assuming that the input has
  // been constant for a long time (compared to the time constant).
  //  LOG(debug) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;

  unsigned short fptc = mTrapConfig->getTrapReg(TrapConfig::kFPTC, mDetector, mRobPos, mMcmPos); // 0..3, 0 - fastest, 3 - slowest

  for (int adc = 0; adc < NADCMCM; adc++) {
    mInternalFilterRegisters[adc].mPedAcc = (baseline << 2) * (1 << mgkFPshifts[fptc]);
  }
  //  LOG(debug) << "LEAVE: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
}

unsigned short TrapSimulator::filterPedestalNextSample(int adc, int timebin, unsigned short value)
{
  // Returns the output of the pedestal filter given the input value.
  // The output depends on the internal registers and, thus, the
  // history of the filter.
  LOG(debug) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__;

  unsigned short fpnp = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos); // 0..511 -> 0..127.75, pedestal at the output
  unsigned short fptc = mTrapConfig->getTrapReg(TrapConfig::kFPTC, mDetector, mRobPos, mMcmPos); // 0..3, 0 - fastest, 3 - slowest
  unsigned short fpby = mTrapConfig->getTrapReg(TrapConfig::kFPBY, mDetector, mRobPos, mMcmPos); // 0..1 bypass, active low

  unsigned short accumulatorShifted;
  unsigned short inpAdd;

  inpAdd = value + fpnp;

  accumulatorShifted = (mInternalFilterRegisters[adc].mPedAcc >> mgkFPshifts[fptc]) & 0x3FF; // 10 bits
  if (timebin == 0)                                                                          // the accumulator is disabled in the drift time
  {
    int correction = (value & 0x3FF) - accumulatorShifted;
    mInternalFilterRegisters[adc].mPedAcc = (mInternalFilterRegisters[adc].mPedAcc + correction) & 0x7FFFFFFF; // 31 bits
  }

  if (fpby == 0) {
    // LOG(info) << "Pedestal filter bypassed";
    return value;
  }
  // LOGF(info, "fpnp(%u), fptc(%u), fpby(%u), inpAdd(%u), accumulatorShifted(%u)", fpnp, fptc, fpby, inpAdd, accumulatorShifted);
  return value; // FIXME bypass hard-coded for now

  if (inpAdd <= accumulatorShifted) {
    return 0;
  } else {
    inpAdd = inpAdd - accumulatorShifted;
    if (inpAdd > 0xFFF) {
      return 0xFFF;
    } else {
      return inpAdd;
    }
  }
}

void TrapSimulator::filterPedestal()
{
  //
  // Apply pedestal filter
  //
  // As the first filter in the chain it reads data from mADCR
  // and outputs to mADCF.
  // It has only an effect if previous samples have been fed to
  // find the pedestal. Currently, the simulation assumes that
  // the input has been stable for a sufficiently long time.
  // LOG(debug) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iAdc = 0; iAdc < NADCMCM; iAdc++) {
      int oldadc = mADCF[iAdc * mNTimeBin + iTimeBin];
      mADCF[iAdc * mNTimeBin + iTimeBin] = filterPedestalNextSample(iAdc, iTimeBin, mADCR[iAdc * mNTimeBin + iTimeBin]);
      //    LOG(debug) << "mADCF : time : " << iTimeBin << " adc : " << iAdc << " change : " << oldadc << " -> " << mADCF[iAdc * mNTimeBin + iTimeBin];
    }
  }
  // LOG(debug) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
}

void TrapSimulator::filterGainInit()
{
  // Initializes the gain filter. In this case, only threshold
  // counters are reset.

  for (int adc = 0; adc < NADCMCM; adc++) {
    // these are counters which in hardware continue
    // until maximum or reset
    mInternalFilterRegisters[adc].mGainCounterA = 0;
    mInternalFilterRegisters[adc].mGainCounterB = 0;
  }
}

unsigned short TrapSimulator::filterGainNextSample(int adc, unsigned short value)
{
  // Apply the gain filter to the given value.
  // BEGIN_LATEX O_{i}(t) = #gamma_{i} * I_{i}(t) + a_{i} END_LATEX
  // The output depends on the internal registers and, thus, the
  // history of the filter.
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " with adc = " << adc << " value = " << value;

  unsigned short mgby = mTrapConfig->getTrapReg(TrapConfig::kFGBY, mDetector, mRobPos, mMcmPos);                             // bypass, active low
  unsigned short mgf = mTrapConfig->getTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGF0 + adc), mDetector, mRobPos, mMcmPos); // 0x700 + (0 & 0x1ff);
  unsigned short mga = mTrapConfig->getTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGA0 + adc), mDetector, mRobPos, mMcmPos); // 40;
  unsigned short mgta = mTrapConfig->getTrapReg(TrapConfig::kFGTA, mDetector, mRobPos, mMcmPos);                             // 20;
  unsigned short mgtb = mTrapConfig->getTrapReg(TrapConfig::kFGTB, mDetector, mRobPos, mMcmPos);                             // 2060;
  //  mgf=256;
  //  mga=8;
  //  mgta=20;
  //  mgtb=2060;

  unsigned int mgfExtended = 0x700 + mgf; // The corr factor which is finally applied has to be extended by 0x700 (hex) or 0.875 (dec)
  // because fgf=0 correspons to 0.875 and fgf=511 correspons to 1.125 - 2^(-11)
  // (see TRAP User Manual for details)
  //if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " with adc = " << adc << " value = " << value << " Trapconfig values :"  << mgby <<":"<<mgf<<":"<<mga<<":"<<mgta<<":"<<mgtb << ":"<< mgfExtended;
  unsigned int corr; // corrected value

  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) << "after declaring corr adc = " << adc << " value = " << value;
  value &= 0xFFF;
  corr = (value * mgfExtended) >> 11;
  corr = corr > 0xfff ? 0xfff : corr;
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr;
  corr = addUintClipping(corr, mga, 12);
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr;

  // Update threshold counters
  // not really useful as they are cleared with every new event
  if (!((mInternalFilterRegisters[adc].mGainCounterA == 0x3FFFFFF) || (mInternalFilterRegisters[adc].mGainCounterB == 0x3FFFFFF)))
  // stop when full
  {
    //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr  << " mgtb : " << mgtb;
    if (corr >= mgtb) {
      mInternalFilterRegisters[adc].mGainCounterB++;
    } else if (corr >= mgta) {
      mInternalFilterRegisters[adc].mGainCounterA++;
    }
  }

  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(debug) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr  << " mgby : " << mgby;
  //  if (mgby == 1)
  //    return corr;
  //  else
  return value;
}

void TrapSimulator::filterGain()
{
  // Read data from mADCF and apply gain filter.

  for (int adc = 0; adc < NADCMCM; adc++) {
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      mADCF[adc * mNTimeBin + iTimeBin] = filterGainNextSample(adc, mADCF[adc * mNTimeBin + iTimeBin]);
    }
  }
}

void TrapSimulator::filterTailInit(int baseline)
{
  // Initializes the tail filter assuming that the input has
  // been at the baseline value (configured by FTFP) for a
  // sufficiently long time.

  // exponents and weight calculated from configuration
  unsigned short alphaLong = 0x3ff & mTrapConfig->getTrapReg(TrapConfig::kFTAL, mDetector, mRobPos, mMcmPos);                            // the weight of the long component
  unsigned short lambdaLong = (1 << 10) | (1 << 9) | (mTrapConfig->getTrapReg(TrapConfig::kFTLL, mDetector, mRobPos, mMcmPos) & 0x1FF);  // the multiplier
  unsigned short lambdaShort = (0 << 10) | (1 << 9) | (mTrapConfig->getTrapReg(TrapConfig::kFTLS, mDetector, mRobPos, mMcmPos) & 0x1FF); // the multiplier

  float lambdaL = lambdaLong * 1.0 / (1 << 11);
  float lambdaS = lambdaShort * 1.0 / (1 << 11);
  float alphaL = alphaLong * 1.0 / (1 << 11);
  float qup, qdn;
  qup = (1 - lambdaL) * (1 - lambdaS);
  qdn = 1 - lambdaS * alphaL - lambdaL * (1 - alphaL);
  float kdc = qup / qdn;

  float ql, qs;

  if (baseline < 0) {
    baseline = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos);
  }

  ql = lambdaL * (1 - lambdaS) * alphaL;
  qs = lambdaS * (1 - lambdaL) * (1 - alphaL);

  for (int adc = 0; adc < NADCMCM; adc++) {
    int value = baseline & 0xFFF;
    int corr = (value * mTrapConfig->getTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGF0 + adc), mDetector, mRobPos, mMcmPos)) >> 11;
    corr = corr > 0xfff ? 0xfff : corr;
    corr = addUintClipping(corr, mTrapConfig->getTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGA0 + adc), mDetector, mRobPos, mMcmPos), 12);

    float kt = kdc * baseline;
    unsigned short aout = baseline - (unsigned short)kt;

    mInternalFilterRegisters[adc].mTailAmplLong = (unsigned short)(aout * ql / (ql + qs));
    mInternalFilterRegisters[adc].mTailAmplShort = (unsigned short)(aout * qs / (ql + qs));
  }
}

unsigned short TrapSimulator::filterTailNextSample(int adc, unsigned short value)
{
  // Returns the output of the tail filter for the given input value.
  // The output depends on the internal registers and, thus, the
  // history of the filter.

  // exponents and weight calculated from configuration
  unsigned short alphaLong = 0x3ff & mTrapConfig->getTrapReg(TrapConfig::kFTAL, mDetector, mRobPos, mMcmPos);                            // the weight of the long component
  unsigned short lambdaLong = (1 << 10) | (1 << 9) | (mTrapConfig->getTrapReg(TrapConfig::kFTLL, mDetector, mRobPos, mMcmPos) & 0x1FF);  // the multiplier of the long component
  unsigned short lambdaShort = (0 << 10) | (1 << 9) | (mTrapConfig->getTrapReg(TrapConfig::kFTLS, mDetector, mRobPos, mMcmPos) & 0x1FF); // the multiplier of the short component

  // intermediate signals
  unsigned int aDiff;
  unsigned int alInpv;
  unsigned short aQ;
  unsigned int tmp;

  unsigned short inpVolt = value & 0xFFF; // 12 bits

  // add the present generator outputs
  aQ = addUintClipping(mInternalFilterRegisters[adc].mTailAmplLong, mInternalFilterRegisters[adc].mTailAmplShort, 12);

  // calculate the difference between the input and the generated signal
  if (inpVolt > aQ) {
    aDiff = inpVolt - aQ;
  } else {
    aDiff = 0;
  }

  // the inputs to the two generators, weighted
  alInpv = (aDiff * alphaLong) >> 11;

  // the new values of the registers, used next time
  // long component
  tmp = addUintClipping(mInternalFilterRegisters[adc].mTailAmplLong, alInpv, 12);
  tmp = (tmp * lambdaLong) >> 11;
  mInternalFilterRegisters[adc].mTailAmplLong = tmp & 0xFFF;
  // short component
  tmp = addUintClipping(mInternalFilterRegisters[adc].mTailAmplShort, aDiff - alInpv, 12);
  tmp = (tmp * lambdaShort) >> 11;
  mInternalFilterRegisters[adc].mTailAmplShort = tmp & 0xFFF;

  // the output of the filter
  if (mTrapConfig->getTrapReg(TrapConfig::kFTBY, mDetector, mRobPos, mMcmPos) == 0) { // bypass mode, active low
    return value;
  } else {
    return aDiff;
  }
}

void TrapSimulator::filterTail()
{
  // Apply tail cancellation filter to all data.

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iAdc = 0; iAdc < NADCMCM; iAdc++) {
      mADCF[iAdc * mNTimeBin + iTimeBin] = filterTailNextSample(iAdc, mADCF[iAdc * mNTimeBin + iTimeBin]);
    }
  }
}

void TrapSimulator::zeroSupressionMapping()
{
  //
  // Zero Suppression Mapping implemented in TRAP chip
  // only implemented for up to 30 timebins
  //
  // See detail TRAP manual "Data Indication" section:
  // http://www.kip.uni-heidelberg.de/ti/TRD/doc/trap/TRAP-UserManual.pdf
  //

  if (!checkInitialized()) {
    return;
  }

  int eBIS = mTrapConfig->getTrapReg(TrapConfig::kEBIS, mDetector, mRobPos, mMcmPos);
  int eBIT = mTrapConfig->getTrapReg(TrapConfig::kEBIT, mDetector, mRobPos, mMcmPos);
  int eBIL = mTrapConfig->getTrapReg(TrapConfig::kEBIL, mDetector, mRobPos, mMcmPos);
  int eBIN = mTrapConfig->getTrapReg(TrapConfig::kEBIN, mDetector, mRobPos, mMcmPos);

  for (int iAdc = 0; iAdc < NADCMCM; iAdc++) {
    mZSMap[iAdc] = -1;
  }

  for (int it = 0; it < mNTimeBin; it++) {
    int iAdc; // current ADC channel
    int ap;
    int ac;
    int an;
    int mask;
    int supp; // suppression of the current channel (low active)

    // ----- first channel -----
    iAdc = 0;

    ap = 0 >> mgkAddDigits;                                  // previous
    ac = mADCF[iAdc * mNTimeBin + it] >> mgkAddDigits;       // current
    an = mADCF[(iAdc + 1) * mNTimeBin + it] >> mgkAddDigits; // next

    mask = (ac >= ap && ac >= an) ? 0 : 0x1; // peak center detection
    mask += (ap + ac + an > eBIT) ? 0 : 0x2; // cluster
    mask += (ac > eBIS) ? 0 : 0x4;           // absolute large peak

    supp = (eBIL >> mask) & 1;

    mZSMap[iAdc] &= ~((1 - supp) << it);
    if (eBIN == 0) { // neighbour sensitivity
      mZSMap[iAdc + 1] &= ~((1 - supp) << it);
    }

    // ----- last channel -----
    iAdc = NADCMCM - 1;

    ap = mADCF[(iAdc - 1) * mNTimeBin + it] >> mgkAddDigits; // previous
    ac = mADCF[iAdc * mNTimeBin + it] >> mgkAddDigits;       // current
    an = 0 >> mgkAddDigits;                                  // next

    mask = (ac >= ap && ac >= an) ? 0 : 0x1; // peak center detection
    mask += (ap + ac + an > eBIT) ? 0 : 0x2; // cluster
    mask += (ac > eBIS) ? 0 : 0x4;           // absolute large peak

    supp = (eBIL >> mask) & 1;

    mZSMap[iAdc] &= ~((1 - supp) << it);
    if (eBIN == 0) { // neighbour sensitivity
      mZSMap[iAdc - 1] &= ~((1 - supp) << it);
    }

    // ----- middle channels -----
    for (iAdc = 1; iAdc < NADCMCM - 1; iAdc++) {
      ap = mADCF[(iAdc - 1) * mNTimeBin + it] >> mgkAddDigits; // previous
      ac = mADCF[iAdc * mNTimeBin + it] >> mgkAddDigits;       // current
      an = mADCF[(iAdc + 1) * mNTimeBin + it] >> mgkAddDigits; // next

      mask = (ac >= ap && ac >= an) ? 0 : 0x1; // peak center detection
      mask += (ap + ac + an > eBIT) ? 0 : 0x2; // cluster
      mask += (ac > eBIS) ? 0 : 0x4;           // absolute large peak

      supp = (eBIL >> mask) & 1;

      mZSMap[iAdc] &= ~((1 - supp) << it);
      if (eBIN == 0) { // neighbour sensitivity
        mZSMap[iAdc - 1] &= ~((1 - supp) << it);
        mZSMap[iAdc + 1] &= ~((1 - supp) << it);
      }
    }
  }
}

void TrapSimulator::addHitToFitreg(int adc, unsigned short timebin, unsigned short qtot, short ypos)
{
  // Add the given hit to the fit register which will be used for
  // the tracklet calculation.
  // In addition to the fit sums in the fit register
  //
  /*
    if ((timebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos)) &&
        (timebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos))) {
      mFitReg[adc].q0 += qtot;
    }

    if ((timebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos)) &&
        (timebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos))) {
      mFitReg[adc].q1 += qtot;
    }
  */
  if (timebin >= 1 && timebin < 6) {
    mFitReg[adc].q0 += qtot;
  }

  if (timebin >= 14 && timebin < 20) {
    mFitReg[adc].q1 += qtot;
  }

  if ((timebin >= 1) &&
      (timebin < 24)) {
    mFitReg[adc].sumX += timebin;
    mFitReg[adc].sumX2 += timebin * timebin;
    mFitReg[adc].nHits++;
    mFitReg[adc].sumY += ypos;
    mFitReg[adc].sumY2 += ypos * ypos;
    mFitReg[adc].sumXY += timebin * ypos;
    LOGF(debug, "FitReg for channel %i and timebin %u: ypos(%i), qtot(%i)", adc, timebin, ypos, qtot);
    // mFitReg.Print();
  }
}

void TrapSimulator::calcFitreg()
{
  // Preprocessing.
  // Detect the hits and fill the fit registers.
  // Requires 12-bit data from mADCF which means Filter()
  // has to be called before even if all filters are bypassed.
  //??? to be clarified:

  // find first timebin to be looked at
  unsigned short timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPFS, mDetector, mRobPos, mMcmPos);
  if (mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos) < timebin1) {
    timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos);
  }
  if (mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos) < timebin1) {
    timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos);
  }

  // find last timebin to be looked at
  unsigned short timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPFE, mDetector, mRobPos, mMcmPos);
  if (mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos) > timebin2) {
    timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos);
  }
  if (mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos) > timebin2) {
    timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos);
  }

  // FIXME: overwrite fit start with values as in Venelin's simulation:
  timebin1 = 1;
  timebin2 = 24;

  // reset the fit registers
  for (auto& fitreg : mFitReg) {
    fitreg.ClearReg();
  }

  for (unsigned int timebin = timebin1; timebin < timebin2; timebin++) {
    // first find the hit candidates and store the total cluster charge in qTotal array
    // in case of not hit store 0 there.
    std::array<unsigned short, 20> qTotal{}; //[19 + 1]; // the last is dummy
    int adcLeft, adcCentral, adcRight;
    for (int adcch = 0; adcch < NADCMCM - 2; adcch++) {
      adcLeft = mADCF[adcch * mNTimeBin + timebin];
      adcCentral = mADCF[(adcch + 1) * mNTimeBin + timebin];
      adcRight = mADCF[(adcch + 2) * mNTimeBin + timebin];
      bool hitQual = false;
      if (mTrapConfig->getTrapReg(TrapConfig::kTPVBY, mDetector, mRobPos, mMcmPos) == 0) {
        // bypass the cluster verification
        hitQual = true;
      } else {
        hitQual = ((adcLeft * adcRight) <
                   ((mTrapConfig->getTrapReg(TrapConfig::kTPVT, mDetector, mRobPos, mMcmPos) * adcCentral * adcCentral) >> 10));
        if (hitQual) {
          LOG(debug) << "cluster quality cut passed with " << adcLeft << ", " << adcCentral << ", "
                     << adcRight << " - threshold " << mTrapConfig->getTrapReg(TrapConfig::kTPVT, mDetector, mRobPos, mMcmPos)
                     << " -> " << mTrapConfig->getTrapReg(TrapConfig::kTPVT, mDetector, mRobPos, mMcmPos) * adcCentral * adcCentral;
        }
      }

      // The accumulated charge is with the pedestal!!!
      int qtotTemp = adcLeft + adcCentral + adcRight;

      if ((hitQual) &&
          (qtotTemp >= mTrapConfig->getTrapReg(TrapConfig::kTPHT, mDetector, mRobPos, mMcmPos)) &&
          (adcLeft <= adcCentral) &&
          (adcCentral > adcRight)) {
        qTotal[adcch] = qtotTemp;
      } else {
        qTotal[adcch] = 0;
      }
    }

    short fromLeft = -1;
    short adcch = 0;
    short found = 0;
    std::array<unsigned short, 6> marked{};
    marked[4] = 19; // invalid channel
    marked[5] = 19; // invalid channel
    qTotal[19] = 0;
    while ((adcch < 16) && (found < 3)) {
      if (qTotal[adcch] > 0) {
        fromLeft = adcch;
        marked[2 * found + 1] = adcch;
        found++;
      }
      adcch++;
    }

    short fromRight = -1;
    adcch = 18;
    found = 0;
    while ((adcch > 2) && (found < 3)) {
      if (qTotal[adcch] > 0) {
        marked[2 * found] = adcch;
        found++;
        fromRight = adcch;
      }
      adcch--;
    }

    // here mask the hit candidates in the middle, if any
    if ((fromLeft >= 0) && (fromRight >= 0) && (fromLeft < fromRight)) {
      for (adcch = fromLeft + 1; adcch < fromRight; adcch++) {
        qTotal[adcch] = 0;
      }
    }

    found = 0;
    for (adcch = 0; adcch < 19; adcch++) {
      if (qTotal[adcch] > 0) {
        found++;
        // NOT READY
      }
    }
    if (found > 4) // sorting like in the TRAP in case of 5 or 6 candidates!
    {
      if (marked[4] == marked[5]) {
        marked[5] = 19;
      }
      std::array<unsigned short, 6> qMarked;
      for (found = 0; found < 6; found++) {
        qMarked[found] = qTotal[marked[found]] >> 4;
      }

      unsigned short worse1, worse2;
      sort6To2Worst(marked[0], marked[3], marked[4], marked[1], marked[2], marked[5],
                    qMarked[0],
                    qMarked[3],
                    qMarked[4],
                    qMarked[1],
                    qMarked[2],
                    qMarked[5],
                    &worse1, &worse2);
      // Now mask the two channels with the smallest charge
      if (worse1 < 19) {
        qTotal[worse1] = 0;
      }
      if (worse2 < 19) {
        qTotal[worse2] = 0;
      }
    }

    for (adcch = 0; adcch < 19; adcch++) {
      if (qTotal[adcch] > 0) // the channel is marked for processing
      {
        adcLeft = mADCF[adcch * mNTimeBin + timebin];
        adcCentral = mADCF[(adcch + 1) * mNTimeBin + timebin];
        adcRight = mADCF[(adcch + 2) * mNTimeBin + timebin];
        LOGF(debug, "ch(%i): left(%i), central(%i), right(%i)", adcch, adcLeft, adcCentral, adcRight);
        //  hit detected, in TRAP we have 4 units and a hit-selection, here we proceed all channels!
        //  subtract the pedestal TPFP, clipping instead of wrapping

        int regTPFP = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos); //TODO put this together with the others as members of trapsim, which is initiliased by det,rob,mcm.
        LOG(debug) << "Hit found, time=" << timebin << ", adcch=" << adcch << "/" << adcch + 1 << "/"
                   << adcch + 2 << ", adc values=" << adcLeft << "/" << adcCentral << "/"
                   << adcRight << ", regTPFP=" << regTPFP << ", TPHT=" << mTrapConfig->getTrapReg(TrapConfig::kTPHT, mDetector, mRobPos, mMcmPos);
        // regTPFP >>= 2; // OS: this line should be commented out when checking real data. It's only needed for comparison with Venelin's simulation if in addition mgkAddDigits == 0
        if (adcLeft < regTPFP) {
          adcLeft = 0;
        } else {
          adcLeft -= regTPFP;
        }
        if (adcCentral < regTPFP) {
          adcCentral = 0;
        } else {
          adcCentral -= regTPFP;
        }
        if (adcRight < regTPFP) {
          adcRight = 0;
        } else {
          adcRight -= regTPFP;
        }

        // Calculate the center of gravity
        // checking for adcCentral != 0 (in case of "bad" configuration)
        if (adcCentral == 0) {
          LOG(error) << "bad configuration detected";
          continue;
        }
        short ypos = 128 * (adcRight - adcLeft) / adcCentral;
        if (ypos < 0) {
          ypos = -ypos;
        }
        // ypos element of [0:128]
        //  make the correction using the position LUT
        // LOG(info) << "ypos raw is " << ypos << "  adcrigh-adcleft/adccentral " << adcRight << "-" << adcLeft << "/" << adcCentral << "==" << (adcRight - adcLeft) / adcCentral << " 128 * numerator : " << 128 * (adcRight - adcLeft) / adcCentral;
        // LOG(info) << "ypos before lut correction : " << ypos;
        ypos = ypos + mTrapConfig->getTrapReg((TrapConfig::TrapReg_t)(TrapConfig::kTPL00 + (ypos & 0x7F)),
                                              mDetector, mRobPos, mMcmPos);
        // ypos += LUT_POS[ypos & 0x7f]; // FIXME use this LUT to obtain the same results as Venelin
        //   LOG(info) << "ypos after lut correction : " << ypos;
        if (adcLeft > adcRight) {
          ypos = -ypos;
        }
        LOGF(debug, "ch(%i): left(%i), central(%i), right(%i), ypos(%i)", adcch, adcLeft, adcCentral, adcRight, ypos);
        addHitToFitreg(adcch, timebin, qTotal[adcch] >> mgkAddDigits, ypos);
      }
    }
  }
}

void TrapSimulator::trackletSelection()
{
  // Select up to 3 tracklet candidates from the fit registers
  // and assign them to the CPUs.
  unsigned short adcIdx, i, j, ntracks, tmp;
  std::array<unsigned short, 18> trackletCandch{};   // store the adcch for all tracklet candidates
  std::array<unsigned short, 18> trackletCandhits{}; // store the number of hits for all tracklet candidates

  ntracks = 0;
  // LOG(info) << "kTPCL: " << mTrapConfig->getTrapReg(TrapConfig::kTPCL, mDetector, mRobPos, mMcmPos);
  // LOG(info) << "kTPCT: " << mTrapConfig->getTrapReg(TrapConfig::kTPCT, mDetector, mRobPos, mMcmPos);
  for (adcIdx = 0; adcIdx < 18; adcIdx++) { // ADCs
    if ((mFitReg[adcIdx].nHits >= mTrapConfig->getTrapReg(TrapConfig::kTPCL, mDetector, mRobPos, mMcmPos)) &&
        (mFitReg[adcIdx].nHits + mFitReg[adcIdx + 1].nHits >= 8)) { // FIXME was 10 otherwise
      trackletCandch[ntracks] = adcIdx;
      trackletCandhits[ntracks] = mFitReg[adcIdx].nHits + mFitReg[adcIdx + 1].nHits;
      //   LOG(debug) << ntracks << " " << trackletCandch[ntracks] << " " << trackletCandhits[ntracks];
      ntracks++;
    };
  }
  LOG(debug) << "Number of track candidates:" << ntracks;
  for (i = 0; i < ntracks; i++) {
    LOG(debug) << "TRACKS: " << i << " " << trackletCandch[i] << " " << trackletCandhits[i];
  }
  if (ntracks > 4) {
    // primitive sorting according to the number of hits
    for (j = 0; j < (ntracks - 1); j++) {
      for (i = j + 1; i < ntracks; i++) {
        if ((trackletCandhits[j] < trackletCandhits[i]) ||
            ((trackletCandhits[j] == trackletCandhits[i]) && (trackletCandch[j] < trackletCandch[i]))) {
          // swap j & i
          tmp = trackletCandhits[j];
          trackletCandhits[j] = trackletCandhits[i];
          trackletCandhits[i] = tmp;
          tmp = trackletCandch[j];
          trackletCandch[j] = trackletCandch[i];
          trackletCandch[i] = tmp;
        }
      }
    }
    ntracks = 4; // cut the rest, 4 is the max
  }
  // else is not necessary to sort

  // now sort, so that the first tracklet going to CPU0 corresponds to the highest adc channel - as in the TRAP
  for (j = 0; j < (ntracks - 1); j++) {
    for (i = j + 1; i < ntracks; i++) {
      if (trackletCandch[j] < trackletCandch[i]) {
        // swap j & i
        tmp = trackletCandhits[j];
        trackletCandhits[j] = trackletCandhits[i];
        trackletCandhits[i] = tmp;
        tmp = trackletCandch[j];
        trackletCandch[j] = trackletCandch[i];
        trackletCandch[i] = tmp;
      }
    }
  }
  for (i = 0; i < ntracks; i++) {   // CPUs with tracklets.
    mFitPtr[i] = trackletCandch[i]; // pointer to the left channel with tracklet for CPU[i]
  }
  for (i = ntracks; i < 4; i++) { // CPUs without tracklets
    mFitPtr[i] = NOTRACKLETFIT;   // pointer to the left channel with tracklet for CPU[i] = NOTRACKLETFIT (invalid)
  }

  // reject multiple tracklets
  // FIXME is now done in fitTracklet() - is this actually optional or always done anyways?
  /*
  if (FeeParam::instance()->getRejectMultipleTracklets()) {
    unsigned short counts = 0;
    for (j = 0; j < (ntracks - 1); j++) {
      if (mFitPtr[j] == NOTRACKLETFIT) {
        continue;
      }

      for (i = j + 1; i < ntracks; i++) {
        // check if tracklets are from neighbouring ADC channels
        if (TMath::Abs(mFitPtr[j] - mFitPtr[i]) > 1.) {
          continue;
        }

        // check which tracklet candidate has higher amount of hits
        if ((mFitReg[mFitPtr[j]].nHits + mFitReg[mFitPtr[j] + 1].nHits) >=
            (mFitReg[mFitPtr[i]].nHits + mFitReg[mFitPtr[i] + 1].nHits)) {
          mFitPtr[i] = NOTRACKLETFIT;
          counts++;
        } else {
          mFitPtr[j] = NOTRACKLETFIT;
          counts++;
          break;
        }
      }
    }

    ntracks = ntracks - counts;
  }
  */
}

void TrapSimulator::fitTracklet()
{
  // Perform the actual tracklet fit based on the fit sums
  // which have been filled in the fit registers.

  std::array<uint32_t, NCPU> adcChannelMask = {0};     // marks the channels which contribute to a tracklet
  std::array<uint32_t, NCPU> charges = {0};            // the charges calculated by each CPU
  std::array<uint16_t, NADCMCM> trap_adc_q2_sum = {0}; // integrated charges in the third window
  int32_t wrks;                                        // signed integer as temporary variable
  uint32_t wrku;                                       // unsigned integer as temporary variable
  for (int cpu = 0; cpu < NCPU - 1; ++cpu) {
    // only cpu0..2 search for tracklets
    if (mFitPtr[cpu] == NOTRACKLETFIT) {
      // 0xFF as 8-bit charge from each CPU in the MCM header is an indicator for missing tracklet
      charges[cpu] = 0xff << (8 * cpu);
      adcChannelMask[cpu] = CHANNELNRNOTRKLT;
    } else {
      adcChannelMask[cpu] = mFitPtr[cpu];
    }
  }
  // eliminate double tracklets
  for (int cpu = 1; cpu < NCPU - 1; ++cpu) {
    // only cpu1..2 search for double tracklets
    if ((mFitPtr[cpu] == mFitPtr[cpu - 1] + 1) || (mFitPtr[cpu] + 1 == mFitPtr[cpu - 1])) {
      // tracklet positions are apart by +1 or -1
      adcChannelMask[cpu] = CHANNELNRNOTRKLT;
      // 0xFF as 8-bit charge from each CPU in the MCM header is an indicator for missing tracklet
      charges[cpu] = 0xff << (8 * cpu);
    }
  }
  for (int cpu = 0; cpu < NCPU; ++cpu) {
    mMCMT[cpu] = TRACKLETENDMARKER;
  }
  // cpu3 prepares the complete ADC channels mask
  // OS: now we mask the left channel which contributes to a tracklet + 3 channels to the right. Should we not take the two neighbouring channels of the tracklet fit?
  wrku = 0xf0; // for one tracklet we mask up to 4 ADC channels
  for (int cpu = 0; cpu < NCPU - 1; ++cpu) {
    wrks = adcChannelMask[cpu];
    wrks -= 4;
    if (wrks > 15) {
      // this indicates that no tracklet is assigned (adcChannelMask[cpu] == CHANNELNRNOTRKLT)
      wrks -= 32; // the shift in TRAP has as argument a 5 bit signed integer!
    }
    if (wrks > 0) {
      adcChannelMask[3] |= wrku << wrks;
    } else {
      adcChannelMask[3] |= wrku >> (-wrks); // has no effect in case tracklet was missing, since wrks = -13 in that case and wrku >> 13 == 0
    }
  }
  // adcChannelMask &= ADC_CHANNEL_MASK_MCM; // FIXME: add internal channel mask for each MCM
  if (adcChannelMask[3] > 0) {
    // there are tracklets, as there are marked ADC channels
    // charge integration in the third window, defined by Q2_LEFT_MRG_VAL, Q2_WIN_WIDTH_VAL
    // each CPU (0..3) has access to a part of the event buffers (0..4, 5..9, 10..14, 15..20 channels)
    for (int ch = 0; ch < NADCMCM; ch++) {
      if ((adcChannelMask[3] >> ch) & 1) {
        // the channel is masked as contributing to a tracklet
        for (int timebin = mQ2LeftMargin; timebin < (mQ2LeftMargin + mQ2WindowWidth); timebin++) {
          trap_adc_q2_sum[ch] += (getDataFiltered(ch, timebin));
          LOGF(debug, "Adding in ch(%i), tb(%i): %i\n", ch, timebin, getDataFiltered(ch, timebin));
        }
      }
      LOGF(debug, "trap_adc_q2_sum[%i]=%i", ch, trap_adc_q2_sum[ch]);
    }

    // begin actual tracklet fit
    int decPlaces = 5;
    int decPlacesSlope = 2;
    int rndAdd = 1 << (decPlaces - 1);

    int yoffs = (21 << (8 + decPlaces)) / 2; // we need to shift to the MCM center

    // add corrections for mis-alignment
    if (FeeParam::instance()->getUseMisalignCorr()) {
      LOG(debug) << "using mis-alignment correction";
      yoffs += (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrYcorr, mDetector, mRobPos, mMcmPos);
    }

    uint64_t shift = 1UL << 32;
    uint64_t scaleY = (shift >> 8) * PADGRANULARITYTRKLPOS;
    uint64_t scaleD = (shift >> 8) * PADGRANULARITYTRKLSLOPE;

    for (int cpu = 0; cpu < 3; cpu++) {
      if (adcChannelMask[cpu] != CHANNELNRNOTRKLT) {
        FitReg* fit0 = &mFitReg[mFitPtr[cpu]];
        FitReg* fit1 = &mFitReg[mFitPtr[cpu] + 1]; // next channel
        // fit0->dumpHex(mFitPtr[cpu]);
        // fit1->dumpHex(mFitPtr[cpu] + 1);

        int64_t mult64 = 1L << (32 + decPlaces);

        // time offset for fit sums
        const int t0 = FeeParam::instance()->getUseTimeOffset() ? (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrTimeOffset, mDetector, mRobPos, mMcmPos) : 0;

        LOG(debug) << "using time offset of t0 = " << t0;

        // Merging
        uint16_t nHits = fit0->nHits + fit1->nHits; // number of hits
        int32_t sumX = fit0->sumX + fit1->sumX;
        int32_t sumX2 = fit0->sumX2 + fit1->sumX2;
        uint32_t q0 = fit0->q0 + fit1->q0;
        uint32_t q1 = fit0->q1 + fit1->q1;
        int32_t sumY = fit0->sumY + fit1->sumY + 256 * fit1->nHits;
        int32_t sumXY = fit0->sumXY + fit1->sumXY + 256 * fit1->sumX;
        int32_t sumY2 = fit0->sumY2 + fit1->sumY2 + 512 * fit1->sumY + 256 * 256 * fit1->nHits; // not used in the current TRAP program, used for error calculation (simulation only)
        LOGF(debug, "Q0(%i), Q1(%i)", q0, q1);

        int32_t denom = nHits * sumX2 - sumX * sumX;
        int32_t mult32 = mult64 / denom; // exactly like in the TRAP program, divide 64 bit to 32 bit and get 32 bit result

        // still without divider here
        int32_t slope = nHits * sumXY - sumX * sumY;
        int32_t position = sumX2 * sumY - sumX * sumXY;

        // first take care of slope
        int64_t temp = slope;
        temp *= mult32;
        slope = temp >> 32;
        // same for offset
        temp = position;
        temp *= mult32;
        position = temp >> 32;
        LOGF(debug, "2 slope=%i, position=%i", slope, position);

        position = position << decPlaces;
        position += t0 * nHits * sumXY - t0 * sumX * sumY;
        position = position >> decPlaces;

        wrks = (mFitPtr[cpu] << (8 + decPlaces)) - yoffs;
        LOGF(debug, "yoffs=%i, wrks=%i", yoffs, wrks);
        position += wrks;

        mult64 = scaleD;
        mult64 *= slope;
        slope = mult64 >> 32; // take the upper 32 bit

        position = -position; // do the inversion as in the TRAP

        mult64 = scaleY;
        mult64 *= position;
        position = mult64 >> 32; // take the upper 32 bit
        LOGF(debug, "3 slope=%i, position=%i", slope, position);
        // rounding, as in the TRAP
        slope = (slope + rndAdd) >> decPlacesSlope;
        position = (position + rndAdd) >> decPlaces;

        slope = -slope; // inversion as for position FIXME: when changed in the actual trap this line can be removed

        LOGF(debug, "  pos =%5d, slope =%5d\n", position, slope);

        // ============> calculation with floating point arithmetic not done in the actual TRAP
        float fitSlope = (float)(nHits * sumXY - sumX * sumY) / (nHits * sumX2 - sumX * sumX);
        float fitOffset = (float)(sumX2 * sumY - sumX * sumXY) / (nHits * sumX2 - sumX * sumX);
        LOGF(debug, "Fit results as float: offset(%f), slope(%f)", fitOffset, fitSlope);
        float sx = (float)sumX;
        float sx2 = (float)sumX2;
        float sy = (float)sumY;
        float sy2 = (float)sumY2;
        float sxy = (float)sumXY;
        float fitError = sy2 - (sx2 * sy * sy - 2 * sx * sxy * sy + nHits * sxy * sxy) / (nHits * sx2 - sx * sx);
        // <========== end calculation with floating point arithmetic

        // lets check boundaries
        if (slope < -128 || slope > 127) {
          LOGF(debug, "Slope is outside of allowed range: %i", slope);
        }
        if (position < -1023) {
          LOGF(warning, "Position is smaller than allowed range (%i), clipping it", position);
          position = -1023;
        }
        if (position > 1023) {
          LOGF(warning, "Position is larger than allowed range (%i), clipping it", position);
          position = 1023;
        }
        // printf("pos=%i, slope=%i\n", position, slope);
        if (slope < -127 || slope > 127) {
          // FIXME put correct boundaries for slope and position in TRAP config?
          LOGF(debug, "Dropping tracklet of CPU %i with slope %i which is out of range", cpu, slope);
          charges[cpu] = 0xff << (8 * cpu);
          mMCMT[cpu] = TRACKLETENDMARKER;
        } else {
          slope = slope & 0xff;
          position = position & 0x7ff;

          // now comes the charge calculation...
          uint32_t q2 = trap_adc_q2_sum[mFitPtr[cpu]] + trap_adc_q2_sum[mFitPtr[cpu] + 1] + trap_adc_q2_sum[mFitPtr[cpu] + 2] + trap_adc_q2_sum[mFitPtr[cpu] + 3]; // from -1 to +2 or from 0 to +3?
          LOGF(debug, "IntCharge of %d ... %d : 0x%04x, 0x%04x 0x%04x, 0x%04x", mFitPtr[cpu], mFitPtr[cpu] + 3, trap_adc_q2_sum[mFitPtr[cpu]], trap_adc_q2_sum[mFitPtr[cpu] + 1], trap_adc_q2_sum[mFitPtr[cpu] + 2], trap_adc_q2_sum[mFitPtr[cpu] + 3]);
          LOGF(debug, "q2 = %i", q2);
          q2 >>= 4; // OS: FIXME understand this factor! If not applied, q2 is usually out of range (> 62)

          if (mUseFloatingPointForQ) {
            // floating point, with 6 bit mantissa and 2 bits for exp
            int shft, shcd;
            wrku = q0 | q1 | q2;
            wrku >>= mDynSize;
            if ((wrku >> mDynShift0) == 0) {
              shft = mDynShift0;
              shcd = 0;
            } else if ((wrku >> mDynShift1) == 0) {
              shft = mDynShift1;
              shcd = 1;
            } else if ((wrku >> mDynShift2) == 0) {
              shft = mDynShift2;
              shcd = 2;
            } else {
              shft = mDynShift3;
              shcd = 3;
            }
            q0 >>= shft;
            q0 &= mDynMask;
            q1 >>= shft;
            q1 &= mDynMask;
            q2 >>= shft;
            q2 &= mDynMask;
            LOGF(debug, "Compressed Q0 %4d, Q1 %4d, Q2 %4d", q0 << shft, q1 << shft, q2 << shft);
            q2 |= shcd << mDynSize;
            if (q2 == mEmptyHPID8) {
              // prevent sending the HPID code for no tracklet
              --q2;
            }
            charges[cpu] = q2;
            charges[cpu] <<= mDynSize;
            charges[cpu] |= q1;
            charges[cpu] <<= mDynSize;
            charges[cpu] |= q0;
          } else {
            // fixed multiplier
            temp = q0; // temporarily move to 64bit variable
            temp *= mScaleQ;
            q0 = temp >> 32;

            temp = q1; // temporarily move to 64bit variable
            temp *= mScaleQ;
            q1 = temp >> 32;

            temp = q2; // temporarily move to 64bit variable
            temp *= mScaleQ;
            q2 = temp >> 32;
            q2 >>= 1; // q2 needs additional shift, since we want the same range as q0 and q1 and q2 is only 6 bit wide

            // clip the charges
            if (q0 > mMaskQ0Q1) {
              q0 = mMaskQ0Q1;
            }
            if (q1 > mMaskQ0Q1) {
              q1 = mMaskQ0Q1;
            }
            if (q2 > mMaxQ2) {
              q2 = mMaxQ2;
            }

            charges[cpu] = q2;
            charges[cpu] <<= mSizeQ0Q1;
            charges[cpu] |= q1;
            charges[cpu] <<= mSizeQ0Q1;
            charges[cpu] |= q0;
          }

          // < pad_position within the MCM (11 bit) | LPID (12 bit) | slope (8 bit) | 0 >
          // the index here is +1, as CPU0 sends the header, CPU1..3 send the tracklets of
          // CPU0..2.
          // the tracklets output is here what exactly sends CPUx
          LOGF(debug, "We have a tracklet! Position(%i), Slope(%i), q0(%i), q1(%i), q2(%i)", position ^ 0x80, slope ^ 0x80, q0, q1, q2);
          // two bits are inverted in order to avoid misinterpretation of tracklet word as end marker
          mMCMT[cpu + 1] = position ^ 0x80;
          mMCMT[cpu + 1] <<= mSizeLPID;
          mMCMT[cpu + 1] |= charges[cpu] & mMaskLPID;
          mMCMT[cpu + 1] <<= 8;
          mMCMT[cpu + 1] |= slope ^ 0x80;
          mMCMT[cpu + 1] <<= 1;
          // here charges is the HPIDx shifted to the proper position (16 for CPU2, 8 for CPU1)
          charges[cpu] >>= 12;
          charges[cpu] <<= 8 * cpu;

          // prepare 64 bit tracklet word directly
          uint64_t trkltWord64 = mTrkltWordEmpty;
          trkltWord64 |= (static_cast<uint64_t>(position & 0x7ff) << Tracklet64::posbs) | (static_cast<uint64_t>(slope & 0xff) << Tracklet64::slopebs) | (q2 << Tracklet64::Q2bs) | (q1 << Tracklet64::Q1bs) | q0;
          mTrackletArray64.emplace_back(trkltWord64);

          // calculate number of hits and MC label
          mTrackletDigitCount.push_back(0);
          for (int ch = 0; ch < NCOLMCM; ch++) { // TODO: check if one should not check each channel instead of each pad?
            if (mADCDigitIndices[ch] >= 0 && ((ch == mFitPtr[cpu]) || (ch == mFitPtr[cpu] + 1))) {
              // we have a digit in one of the two channels which were used to fit the tracklet
              mTrackletDigitCount.back() += 1;
              mTrackletDigitIndices.push_back(mADCDigitIndices[ch]);
            }
          }
        } // tracklet did pass slope selection
      }   // there is a tracklet available for this CPU
    }     // loop over CPUs for tracklet calculation
  }       // tracklet are available for at least one CPU for this MCM

  // mMCMT[0] stores the MCM tracklet header, the part without HPIDs (mMcmHeaderEmpty) prepared already
  // < 1 | padrow (4 bits) | column (2 bits) | HPID2 (8 bit) | HPID1 (8 bit) | HPID0 (8 bit) | 1 >
  wrku = charges[0] | charges[1] | charges[2];
  // send only if at least one of the HPIDs different from the empty one OR
  // the sending of header without tracklets is not disabled
  if ((wrku != mEmptyHPID24) || !mDontSendEmptyHeaderTrklt) // send MCM header
  {
    mMCMT[0] = mMcmHeaderEmpty | (wrku << 1);
  }

  LOG(debug) << "4x32 bit tracklet data:";
  for (int i = 0; i < 4; ++i) {
    LOGF(debug, "0x%08x", mMCMT[i]);
  }
}

void TrapSimulator::tracklet()
{
  // Run the tracklet calculation by calling sequentially:
  // calcFitreg(); trackletSelection(); fitTracklet()
  // and store the tracklets

  if (!mInitialized) {
    return;
  }

  calcFitreg();
  trackletSelection();
  fitTracklet();
}

// help functions, to be cleaned up

unsigned int TrapSimulator::addUintClipping(unsigned int a, unsigned int b, unsigned int nbits) const
{
  //
  // This function adds a and b (unsigned) and clips to
  // the specified number of bits.
  //

  unsigned int sum = a + b;
  if (nbits < 32) {
    unsigned int maxv = (1 << nbits) - 1;
    if (sum > maxv) {
      sum = maxv;
    }
  } else {
    if ((sum < a) || (sum < b)) {
      sum = 0xFFFFFFFF;
    }
  }
  return sum;
}

void TrapSimulator::sort2(uint16_t idx1i, uint16_t idx2i,
                          uint16_t val1i, uint16_t val2i,
                          uint16_t* idx1o, uint16_t* idx2o,
                          uint16_t* val1o, uint16_t* val2o) const
{
  // sorting for tracklet selection

  if (val1i > val2i) {
    *idx1o = idx1i;
    *idx2o = idx2i;
    *val1o = val1i;
    *val2o = val2i;
  } else {
    *idx1o = idx2i;
    *idx2o = idx1i;
    *val1o = val2i;
    *val2o = val1i;
  }
}

void TrapSimulator::sort3(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i,
                          uint16_t val1i, uint16_t val2i, uint16_t val3i,
                          uint16_t* idx1o, uint16_t* idx2o, uint16_t* idx3o,
                          uint16_t* val1o, uint16_t* val2o, uint16_t* val3o) const
{
  // sorting for tracklet selection

  int sel;

  if (val1i > val2i) {
    sel = 4;
  } else {
    sel = 0;
  }
  if (val2i > val3i) {
    sel = sel + 2;
  }
  if (val3i > val1i) {
    sel = sel + 1;
  }
  switch (sel) {
    case 6: // 1 >  2  >  3            => 1 2 3
    case 0: // 1 =  2  =  3            => 1 2 3 : in this case doesn't matter, but so is in hardware!
      *idx1o = idx1i;
      *idx2o = idx2i;
      *idx3o = idx3i;
      *val1o = val1i;
      *val2o = val2i;
      *val3o = val3i;
      break;

    case 4: // 1 >  2, 2 <= 3, 3 <= 1  => 1 3 2
      *idx1o = idx1i;
      *idx2o = idx3i;
      *idx3o = idx2i;
      *val1o = val1i;
      *val2o = val3i;
      *val3o = val2i;
      break;

    case 2: // 1 <= 2, 2 > 3, 3 <= 1   => 2 1 3
      *idx1o = idx2i;
      *idx2o = idx1i;
      *idx3o = idx3i;
      *val1o = val2i;
      *val2o = val1i;
      *val3o = val3i;
      break;

    case 3: // 1 <= 2, 2 > 3, 3  > 1   => 2 3 1
      *idx1o = idx2i;
      *idx2o = idx3i;
      *idx3o = idx1i;
      *val1o = val2i;
      *val2o = val3i;
      *val3o = val1i;
      break;

    case 1: // 1 <= 2, 2 <= 3, 3 > 1   => 3 2 1
      *idx1o = idx3i;
      *idx2o = idx2i;
      *idx3o = idx1i;
      *val1o = val3i;
      *val2o = val2i;
      *val3o = val1i;
      break;

    case 5: // 1 > 2, 2 <= 3, 3 >  1   => 3 1 2
      *idx1o = idx3i;
      *idx2o = idx1i;
      *idx3o = idx2i;
      *val1o = val3i;
      *val2o = val1i;
      *val3o = val2i;
      break;

    default: // the rest should NEVER happen!
      LOG(error) << "ERROR in Sort3!!!";
      break;
  }
}

void TrapSimulator::sort6To4(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i, uint16_t idx4i, uint16_t idx5i, uint16_t idx6i,
                             uint16_t val1i, uint16_t val2i, uint16_t val3i, uint16_t val4i, uint16_t val5i, uint16_t val6i,
                             uint16_t* idx1o, uint16_t* idx2o, uint16_t* idx3o, uint16_t* idx4o,
                             uint16_t* val1o, uint16_t* val2o, uint16_t* val3o, uint16_t* val4o) const
{
  // sorting for tracklet selection

  uint16_t idx21s, idx22s, idx23s, dummy;
  uint16_t val21s, val22s, val23s;
  uint16_t idx23as, idx23bs;
  uint16_t val23as, val23bs;

  sort3(idx1i, idx2i, idx3i, val1i, val2i, val3i,
        idx1o, &idx21s, &idx23as,
        val1o, &val21s, &val23as);

  sort3(idx4i, idx5i, idx6i, val4i, val5i, val6i,
        idx2o, &idx22s, &idx23bs,
        val2o, &val22s, &val23bs);

  sort2(idx23as, idx23bs, val23as, val23bs, &idx23s, &dummy, &val23s, &dummy);

  sort3(idx21s, idx22s, idx23s, val21s, val22s, val23s,
        idx3o, idx4o, &dummy,
        val3o, val4o, &dummy);
}

void TrapSimulator::sort6To2Worst(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i, uint16_t idx4i, uint16_t idx5i, uint16_t idx6i,
                                  uint16_t val1i, uint16_t val2i, uint16_t val3i, uint16_t val4i, uint16_t val5i, uint16_t val6i,
                                  uint16_t* idx5o, uint16_t* idx6o) const
{
  // sorting for tracklet selection

  uint16_t idx21s, idx22s, idx23s, dummy1, dummy2, dummy3, dummy4, dummy5;
  uint16_t val21s, val22s, val23s;
  uint16_t idx23as, idx23bs;
  uint16_t val23as, val23bs;

  sort3(idx1i, idx2i, idx3i, val1i, val2i, val3i,
        &dummy1, &idx21s, &idx23as,
        &dummy2, &val21s, &val23as);

  sort3(idx4i, idx5i, idx6i, val4i, val5i, val6i,
        &dummy1, &idx22s, &idx23bs,
        &dummy2, &val22s, &val23bs);

  sort2(idx23as, idx23bs, val23as, val23bs, &idx23s, idx5o, &val23s, &dummy1);

  sort3(idx21s, idx22s, idx23s, val21s, val22s, val23s,
        &dummy1, &dummy2, idx6o,
        &dummy3, &dummy4, &dummy5);
}

bool TrapSimulator::readPackedConfig(TrapConfig* cfg, int hc, unsigned int* data, int size)
{
  // Read the packed configuration from the passed memory block
  //
  // To be used to retrieve the TRAP configuration from the
  // configuration as sent in the raw data.

  LOG(debug) << "Reading packed configuration";

  int det = hc / 2;

  int idx = 0;
  int err = 0;
  int step, bwidth, nwords, exitFlag, bitcnt;

  unsigned short caddr;
  unsigned int dat, msk, header, dataHi;

  while (idx < size && *data != 0x00000000) {

    int rob = (*data >> 28) & 0x7;
    int mcm = (*data >> 24) & 0xf;

    LOG(debug) << "Config of det. " << det << " MCM " << rob << ":" << mcm << " (0x" << std::hex << *data << ")";
    data++;

    while (idx < size && *data != 0x00000000) {

      header = *data;
      data++;
      idx++;

      LOG(debug) << "read: 0x" << hex << header;

      if (header & 0x01) // single data
      {
        dat = (header >> 2) & 0xFFFF;    // 16 bit data
        caddr = (header >> 18) & 0x3FFF; // 14 bit address

        if (caddr != 0x1FFF) // temp!!! because the end marker was wrong
        {
          if (header & 0x02) // check if > 16 bits
          {
            dataHi = *data;
            LOG(debug) << "read: 0x" << hex << dataHi;
            data++;
            idx++;
            err += ((dataHi ^ (dat | 1)) & 0xFFFF) != 0;
            dat = (dataHi & 0xFFFF0000) | dat;
          }
          LOG(debug) << "addr=0x" << hex << caddr << "(" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ") data=0x" << hex << dat;
          if (!cfg->poke(caddr, dat, det, rob, mcm)) {
            LOG(debug) << "(single-write): non-existing address 0x" << std::hex << caddr << " containing 0x" << std::hex << header;
          }
          if (idx > size) {
            LOG(debug) << "(single-write): no more data, missing end marker";
            return -err;
          }
        } else {
          LOG(debug) << "(single-write): address 0x" << setw(4) << std::hex << caddr << " => old endmarker?" << std::dec;
          return err;
        }
      }

      else // block of data
      {
        step = (header >> 1) & 0x0003;
        bwidth = ((header >> 3) & 0x001F) + 1;
        nwords = (header >> 8) & 0x00FF;
        caddr = (header >> 16) & 0xFFFF;
        exitFlag = (step == 0) || (step == 3) || (nwords == 0);

        if (exitFlag) {
          break;
        }

        switch (bwidth) {
          case 15:
          case 10:
          case 7:
          case 6:
          case 5: {
            msk = (1 << bwidth) - 1;
            bitcnt = 0;
            while (nwords > 0) {
              nwords--;
              bitcnt -= bwidth;
              if (bitcnt < 0) {
                header = *data;
                LOG(debug) << "read 0x" << setw(8) << std::hex << header << std::dec;
                data++;
                idx++;
                err += (header & 1);
                header = header >> 1;
                bitcnt = 31 - bwidth;
              }
              LOG(debug) << "addr=0x" << setw(4) << std::hex << caddr << "(" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ") data=0x" << setw(8) << std::hex << (header & msk);
              if (!cfg->poke(caddr, header & msk, det, rob, mcm)) {
                LOG(debug) << "(single-write): non-existing address 0x" << setw(4) << std::hex << caddr << " containing 0x" << setw(8) << std::hex << header << std::dec;
              }

              caddr += step;
              header = header >> bwidth;
              if (idx >= size) {
                LOG(debug) << "(block-write): no end marker! " << idx << " words read";
                return -err;
              }
            }
            break;
          } // end case 5-15
          case 31: {
            while (nwords > 0) {
              header = *data;
              LOG(debug) << "read 0x" << setw(8) << std::hex << header;
              data++;
              idx++;
              nwords--;
              err += (header & 1);

              LOG(debug) << "addr=0x" << hex << setw(4) << caddr << " (" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ")  data=0x" << hex << setw(8) << (header >> 1);
              if (!cfg->poke(caddr, header >> 1, det, rob, mcm)) {
                LOG(debug) << "(single-write): non-existing address 0x" << setw(4) << std::hex << " containing 0x" << setw(8) << std::hex << header << std::dec;
              }

              caddr += step;
              if (idx >= size) {
                LOG(debug) << "no end marker! " << idx << " words read";
                return -err;
              }
            }
            break;
          }
          default:
            return err;
        } // end switch
      }   // end block case
    }
  } // end while
  LOG(debug) << "no end marker! " << idx << " words read";
  return -err; // only if the max length of the block reached!
}
