// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD MCM (Multi Chip Module) simulator                                    //
//  which simulates the TRAP processing after the AD-conversion.             //
//  The relevant parameters (i.e. configuration settings of the TRAP)        //
//  are taken from TrapConfig.                                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

//#include "TTreeStream.h"

#include "TRDBase/TRDSimParam.h"
#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/FeeParam.h"
#include "TRDBase/Tracklet.h"
#include "TRDBase/CalOnlineGainTables.h"
#include "TRDSimulation/TrapConfigHandler.h"
#include "TRDSimulation/TrapConfig.h"
#include "TRDSimulation/TrapSimulator.h"
#include "fairlogger/Logger.h"

//to pull in the digitzer incomnig data.
#include "TRDBase/Digit.h"
#include "TRDSimulation/Digitizer.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>

#include <iostream>
#include <iomanip>
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TLine.h"
#include "TRandom.h"
#include "TMath.h"
#include <TTree.h>
#include <TFile.h>
#include "TRandom3.h"
#include <ostream>
#include <fstream>

using namespace o2::trd;
using namespace std;

#define DEBUGTRAP 1

bool TrapSimulator::mgApplyCut = true;
int TrapSimulator::mgAddBaseline = 0;
bool TrapSimulator::mgStoreClusters = true;
const int TrapSimulator::mgkFormatIndex = std::ios_base::xalloc();
const std::array<unsigned short, 4> TrapSimulator::mgkFPshifts{11, 14, 17, 21};

TrapSimulator::TrapSimulator()
  : mInitialized(false), mDetector(-1), mRobPos(-1), mMcmPos(-1), mRow(-1), mNTimeBin(-1), mTrklBranchName("mcmtrklbranch"), mFeeParam(nullptr), mTrapConfig(nullptr)
{
  //
  // TrapSimulator default constructor
  // By default, nothing is initialized.
  // It is necessary to issue init before use.

  mFitPtr[0] = 0;
  mFitPtr[1] = 0;
  mFitPtr[2] = 0;
  mFitPtr[3] = 0;
  mNHits = 0;
  //  mCalib.setCCDBForSimulation(297595);
}

void TrapSimulator::init(TrapConfig* trapconfig, int det, int robPos, int mcmPos)
{
  //
  // Initialize the class with new MCM position information
  // memory is allocated in the first initialization
  //
  //
  LOG(debug4) << " : trap sim is at : 0x" << hex << trapconfig;
  if (!mInitialized) {
    mFeeParam = FeeParam::instance();
    mTrapConfig = trapconfig;
  }

  mDetector = det;
  mRobPos = robPos;
  mMcmPos = mcmPos;
  mRow = mFeeParam->getPadRowFromMCM(mRobPos, mMcmPos);

  if (!mInitialized) {
    mNTimeBin = mTrapConfig->getTrapReg(TrapConfig::kC13CPUA, mDetector, mRobPos, mMcmPos);
    mZSMap.resize(FeeParam::getNadcMcm());

    // tracklet calculation
    //now a 25 slot array mFitReg.resize(FeeParam::getNadcMcm()); //TODO for now this is constant size in an array not a vector
    //  mTrackletArray-.resize(mgkMaxTracklets);
    //now a 100 slot array as per run2  mHits.resize(50);
    mMCMT.resize(mgkMaxTracklets);

    mADCR.resize(mNTimeBin * FeeParam::getNadcMcm());
    mADCF.resize(mNTimeBin * FeeParam::getNadcMcm());
  }

  mInitialized = true;

  mNHits = 0;

  reset();
}

void TrapSimulator::reset()
{
  // Resets the data values and internal filter registers
  // by re-initialising them

  if (!checkInitialized())
    return;

  //clear the adc data
  memset(&mADCR[0], 0, sizeof(mADCR[0]) * mADCR.size());
  memset(&mADCF[0], 0, sizeof(mADCF[0]) * mADCF.size());
  //clear the labels
  mADCLabels[0].clear();
  mADCLabels[1].clear();
  mADCLabels[2].clear();

  mTrackletLabels.clear(); // as the name implies clear the stored labels
  mNHits = 0;

  for (auto filterreg : mInternalFilterRegisters)
    filterreg.ClearReg();
  // Default unread, low active bit mask
  memset(&mZSMap[0], 0, sizeof(mZSMap[0]) * FeeParam::getNadcMcm());
  memset(&mMCMT[0], 0, sizeof(mMCMT[0]) * mgkMaxTracklets);
  //mDict1.clear();
  //mDict2.clear();
  //mDict3.clear();

  filterPedestalInit();
  filterGainInit();
  filterTailInit();
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
  if (!mcm.checkInitialized())
    return os;

  // ----- human-readable output -----
  if (os.iword(TrapSimulator::mgkFormatIndex) == 0) {

    os << "TRAP " << mcm.getMcmPos() << " on ROB " << mcm.getRobPos() << " in detector " << mcm.getDetector() << std::endl;

    os << "----- Unfiltered ADC data (10 bit) -----" << std::endl;
    os << "ch    ";
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++)
      os << std::setw(5) << iChannel;
    os << std::endl;
    for (int iTimeBin = 0; iTimeBin < mcm.getNumberOfTimeBins(); iTimeBin++) {
      os << "tb " << std::setw(2) << iTimeBin << ":";
      for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
        os << std::setw(5) << (mcm.getDataRaw(iChannel, iTimeBin) >> mcm.mgkAddDigits);
      }
      os << std::endl;
    }
    os << "----- Filtered ADC data (10+2 bit) -----" << std::endl;
    os << "ch    ";
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++)
      os << std::setw(4) << iChannel
         << ((~mcm.getZeroSupressionMap(iChannel) != 0) ? "!" : " ");
    os << std::endl;
    for (int iTimeBin = 0; iTimeBin < mcm.getNumberOfTimeBins(); iTimeBin++) {
      os << "tb " << std::setw(2) << iTimeBin << ":";
      for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
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
      for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
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
    buf.resize(bufSize);

    int bufLength = mcm.produceRawStream(buf);

    for (int i = 0; i < bufLength; i++)
      std::cout << "0x" << std::hex << buf[i] << std::dec << std::endl;

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
    if (mFitPtr[cpu] != 31)
      tracklet = true;
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
      if (mFitPtr[cpu] != 31) {
        for (int adcch = mFitPtr[cpu]; adcch < mFitPtr[cpu] + 2; adcch++) {
          if (adcch > 24) {
            LOG(error) << "adcch going awol : " << adcch << " > 25";
          }
          os << "    <ch chnr=\"" << adcch << "\">" << std::endl;
          os << "     <hits>" << mFitReg[adcch].mNhits << "</hits>" << std::endl;
          os << "     <q0>" << mFitReg[adcch].mQ0 << "</q0>" << std::endl;
          os << "     <q1>" << mFitReg[adcch].mQ1 << "</q1>" << std::endl;
          os << "     <sumx>" << mFitReg[adcch].mSumX << "</sumx>" << std::endl;
          os << "     <sumxsq>" << mFitReg[adcch].mSumX2 << "</sumxsq>" << std::endl;
          os << "     <sumy>" << mFitReg[adcch].mSumY << "</sumy>" << std::endl;
          os << "     <sumysq>" << mFitReg[adcch].mSumY2 << "</sumysq>" << std::endl;
          os << "     <sumxy>" << mFitReg[adcch].mSumXY << "</sumxy>" << std::endl;
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

  const char* const cmdid{R"(" cmdid="0">)"};
  os << "<nginject>" << std::endl;
  os << "<ack roc=\"" << mDetector << cmdid << std::endl;
  os << "<dmem-readout>" << std::endl;
  os << "<d det=\"" << mDetector << "\">" << std::endl;
  os << "  <ro-board rob=\"" << mRobPos << "\">" << std::endl;
  os << "    <m mcm=\"" << mMcmPos << "\">" << std::endl;

  int pid, padrow, slope, offset;
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
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); ++iChannel) {
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
  for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++)
    os << std::setw(5) << iChannel;
  os << std::endl;
  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    os << "tb " << std::setw(2) << iTimeBin << ":";
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
      os << std::setw(5) << (getDataRaw(iChannel, iTimeBin) >> mgkAddDigits);
    }
    os << std::endl;
  }

  os << "----- Filtered ADC data (10+2 bit) -----" << std::endl;
  os << "ch    ";
  for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++)
    os << std::setw(4) << iChannel
       << ((~mZSMap[iChannel] != 0) ? "!" : " ");
  os << std::endl;
  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    os << "tb " << std::setw(2) << iTimeBin << ":";
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
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

  for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
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
    for (int iChannel = 0; iChannel < FeeParam::getNadcMcm(); iChannel++) {
      if ((iTimeBin < timeBinOffset) || (iTimeBin >= mNTimeBin + timeBinOffset)) {
        if (broadcast == false)
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, 10, getRobPos(), getMcmPos());
        else
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, 10, 0, 127);
      } else {
        if (broadcast == false)
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, (getDataFiltered(iChannel, iTimeBin - timeBinOffset) / 4), getRobPos(), getMcmPos());
        else
          mTrapConfig->printDatx(os, addrOffset + iChannel * addrStep + addrOffsetEBSIA + iTimeBin, (getDataFiltered(iChannel, iTimeBin - timeBinOffset) / 4), 0, 127);
      }
    }
    os << std::endl;
  }
}

void TrapSimulator::printPidLutHuman()
{
  // print PID LUT in human readable format

  unsigned int addrEnd = mgkDmemAddrLUTStart + mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTLength, mDetector, mRobPos, mMcmPos) / 4; // /4 because each addr contains 4 values
  unsigned int nBinsQ0 = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTnbins, mDetector, mRobPos, mMcmPos);

  std::cout << "nBinsQ0: " << nBinsQ0 << std::endl;
  std::cout << "LUT table length: " << mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTLength, mDetector, mRobPos, mMcmPos) << std::endl;

  if (nBinsQ0 > 0) {
    for (unsigned int addr = mgkDmemAddrLUTStart; addr < addrEnd; addr++) {
      unsigned int result;
      result = mTrapConfig->getDmemUnsigned(addr, mDetector, mRobPos, mMcmPos);
      std::cout << addr << " # x: " << ((addr - mgkDmemAddrLUTStart) % ((nBinsQ0) / 4)) * 4 << ", y: " << (addr - mgkDmemAddrLUTStart) / (nBinsQ0 / 4)
                << "  #  " << ((result >> 0) & 0xFF)
                << " | " << ((result >> 8) & 0xFF)
                << " | " << ((result >> 16) & 0xFF)
                << " | " << ((result >> 24) & 0xFF) << std::endl;
    }
  }
}

void TrapSimulator::setNTimebins(int ntimebins)
{
  // Reallocate memory if a change in the number of timebins
  // is needed (should not be the case for real data)

  LOG(fatal) << "setNTimebins(" << ntimebins << ") not implemented as we can no longer change the size of the ADC array";
  if (!checkInitialized())
    return;

  mNTimeBin = ntimebins;
  // for( int iAdc = 0 ; iAdc < FeeParam::getNadcMcm(); iAdc++ ) {
  //  delete [] mADCR[iAdc];
  //  delete [] mADCF[iAdc];
  //  mADCR[iAdc] = new int[mNTimeBin];
  //  mADCF[iAdc] = new int[mNTimeBin];
  // }
}

bool TrapSimulator::loadMCM(int det, int rob, int mcm)
{
  // loads the ADC data as obtained from the digitsManager for the specified MCM.
  // This method is meant for rare execution, e.g. in the visualization. When called
  // frequently use SetData(...) instead.
  LOG(fatal) << "no longer implemented, this is left incase script calls it that I have not found yet.";

  return false;
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

  if (!checkInitialized())
    return;

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

    if (inputGain == 0)
      valueg = filterGainNextSample(1, ((int)value) << 2);
    else
      valueg = filterGainNextSample(1, valuep);

    if (inputTail == 0)
      valuet = filterTailNextSample(1, ((int)value) << 2);
    else if (inputTail == 1)
      valuet = filterTailNextSample(1, valuep);
    else
      valuet = filterTailNextSample(1, valueg);

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

bool TrapSimulator::checkInitialized() const
{
  //
  // Check whether object is initialized
  //

  //  if (!mInitialized)
  //   LOG(debug4) << "TrapSimulator is not initialized but function other than Init() is called.";

  return mInitialized;
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

  if (!checkInitialized())
    return;

  LOG(info) << "MCM " << mMcmPos << "  on ROB " << mRobPos << " in detector " << mDetector;

  //std::string opt = option;
  if ((choice & PRINTRAW) != 0 || (choice & PRINTFILTERED) != 0) {
    std::cout << *this;
  }

  if ((choice & PRINTDETECTED) != 0) {
    LOG(info) << "Found " << mNHits << " hits:";
    for (int iHit = 0; iHit < mNHits; iHit++) {
      LOG(info) << "Hit " << std::setw(3) << iHit << " in timebin " << std::setw(2) << mHits[iHit].mTimebin << ", ADC " << std::setw(2) << mHits[iHit].mChannel << " has charge " << std::setw(3) << mHits[iHit].mQtot << " and position " << mHits[iHit].mYpos;
    }
  }

  if ((choice & PRINTFOUND) != 0) {
    LOG(info) << "Found Tracklets:";
    for (int iTrkl = 0; iTrkl < mTrackletArray.size(); iTrkl++) {
      LOG(info) << "tracklet " << iTrkl << ": 0x" << hex << std::setw(8) << mTrackletArray[iTrkl].getTrackletWord();
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
  if (!checkInitialized())
    return;
  TFile* rootfile = new TFile("trdtrackletplots.root", "UPDATE");
  TCanvas* c1 = new TCanvas(Form("canvas_%i_%i:%i:%i_%i", index, mDetector, mRobPos, mMcmPos, (int)mTrackletArray.size()));
  TH2F* hist = new TH2F(Form("mcmdata_%i", index),
                        Form("Data of MCM %i on ROB %i in detector %i ", mMcmPos, mRobPos, mDetector),
                        FeeParam::getNadcMcm(),
                        -0.5,
                        FeeParam::getNadcMcm() - 0.5,
                        getNumberOfTimeBins(),
                        -0.5,
                        getNumberOfTimeBins() - 0.5);
  hist->GetXaxis()->SetTitle("ADC Channel");
  hist->GetYaxis()->SetTitle("Timebin");
  hist->SetStats(false);
  TH2F* histfiltered = new TH2F(Form("mcmdataf_%i", index),
                                Form("Data of MCM %i on ROB %i in detector %i filtered", mMcmPos, mRobPos, mDetector),
                                FeeParam::getNadcMcm(),
                                -0.5,
                                FeeParam::getNadcMcm() - 0.5,
                                getNumberOfTimeBins(),
                                -0.5,
                                getNumberOfTimeBins() - 0.5);
  histfiltered->GetXaxis()->SetTitle("ADC Channel");
  histfiltered->GetYaxis()->SetTitle("Timebin");

  if ((choice & PLOTRAW) != 0) {
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
        hist->SetBinContent(iAdc + 1, iTimeBin + 1, mADCR[iAdc * mNTimeBin + iTimeBin] >> mgkAddDigits);
      }
    }
    hist->Draw("COLZ");
  } else {
    for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
      for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
        histfiltered->SetBinContent(iAdc + 1, iTimeBin + 1, mADCF[iAdc * mNTimeBin + iTimeBin] >> mgkAddDigits);
      }
    }
    histfiltered->Draw("COLZ");
  }

  if ((choice & PLOTHITS) != 0) {
    TGraph* grHits = new TGraph();
    for (int iHit = 0; iHit < mNHits; iHit++) {
      grHits->SetPoint(iHit,
                       mHits[iHit].mChannel + 1 + mHits[iHit].mYpos / 256.,
                       mHits[iHit].mTimebin);
    }
    grHits->Draw("*");
  }

  if ((choice & PLOTTRACKLETS) != 0) {
    TLine* trklLines = new TLine[4];
    LOG(info) << "Tracklet start for index : " << index;
    if (mTrackletArray.size() > 0)
      LOG(info) << "Tracklet : for " << mTrackletArray[0].getDetector() << "::" << mTrackletArray[0].getROB() << " : " << mTrackletArray[0].getMCM();
    else
      LOG(info) << "Tracklet : for trackletarray size of zero ";
    for (int iTrkl = 0; iTrkl < mTrackletArray.size(); iTrkl++) {
      Tracklet trkl = mTrackletArray[iTrkl];
      float padWidth = 0.635 + 0.03 * (mDetector % 6);
      float offset = padWidth / 256. * ((((((mRobPos & 0x1) << 2) + (mMcmPos & 0x3)) * 18) << 8) - ((18 * 4 * 2 - 18 * 2 - 3) << 7)); // revert adding offset in FitTracklet
                                                                                                                                      //TODO replace the 18, 4 3 and 7 with constants for readability
      int ndrift = mTrapConfig->getDmemUnsigned(mgkDmemAddrNdrift, mDetector, mRobPos, mMcmPos) >> 5;
      float slope = 0;
      if (ndrift)
        slope = trkl.getdY() * 140e-4 / ndrift;

      int t0 = mTrapConfig->getTrapReg(TrapConfig::kTPFS, mDetector, mRobPos, mMcmPos);
      int t1 = mTrapConfig->getTrapReg(TrapConfig::kTPFE, mDetector, mRobPos, mMcmPos);

      trklLines[iTrkl].SetX1((offset - (trkl.getY() - slope * t0)) / padWidth); // ??? sign?
      trklLines[iTrkl].SetY1(t0);
      trklLines[iTrkl].SetX2((offset - (trkl.getY() - slope * t1)) / padWidth); // ??? sign?
      trklLines[iTrkl].SetY2(t1);
      trklLines[iTrkl].SetLineColor(2);
      trklLines[iTrkl].SetLineWidth(2);
      LOG(info) << "Tracklet " << iTrkl << ": y = " << trkl.getY() << ", dy = " << (((float)trkl.getdY()) * 140e-4) << " offset : " << offset << "for a det:rob:mcm combo of : " << mDetector << ":" << mRobPos << ":" << mMcmPos;
      LOG(info) << "Tracklet " << iTrkl << ": x1,y1,x2,y2 :: " << trklLines[iTrkl].GetX1() << "," << trklLines[iTrkl].GetY1() << "," << trklLines[iTrkl].GetX2() << "," << trklLines[iTrkl].GetY2();
      LOG(info) << "Tracklet " << iTrkl << ": t0 : " << t0 << ", t1 " << t1 << ", padwidth:" << padWidth << ", slope:" << slope << ", ndrift:" << ndrift << " which comes from : " << mTrapConfig->getDmemUnsigned(mgkDmemAddrNdrift, mDetector, mRobPos, mMcmPos) << " shifted 5 to the right ";
      trklLines[iTrkl].Draw();
    }
    LOG(info) << "Tracklet end ...";
  }
  c1->Write();
  rootfile->Close();
}

void TrapSimulator::setData(int adc, const ArrayADC& data, std::vector<o2::MCCompLabel>& labels)
{
  //
  // Store ADC data into array of raw data
  //

  if (!checkInitialized())
    return;

  if (adc < 0 || adc >= FeeParam::getNadcMcm()) {
    //    LOG(error) << "Error: ADC " << adc << " is out of range (0 .. " << FeeParam::getNadcMcm() - 1 << ")";
    return;
  }

  //  LOG(info) <<  "Set Data : Det:Rob:MCM::"<< getDetector() <<":" <<getRobPos()<<":"<<getMcmPos() << " t:"<< mNTimeBin;
  for (int it = 0; it < mNTimeBin; it++) {
    mADCR[adc * mNTimeBin + it] = ((int)(data[it]) << mgkAddDigits) + (mgAddBaseline << mgkAddDigits);
    mADCF[adc * mNTimeBin + it] = ((int)(data[it]) << mgkAddDigits) + (mgAddBaseline << mgkAddDigits);
    // mADCR[adc * mNTimeBin + it] = (unsigned int)(data[it]);
    // mADCF[adc * mNTimeBin + it] = (unsigned int)(data[it]);
    //    LOG(info) << data[it] <<" at "<< adc << "*" << mNTimeBin <<"+"<<it <<"="<<  adc*mNTimeBin+it << " with data :"<<mADCR[adc*mNTimeBin+it] << ":" << mADCF[adc*mNTimeBin+it];
  }
  mDataIsSet = true;
  mADCFilled |= (1 << adc);
  //for (auto& tmplabel : labels) mADCLabels[adc].push_back(tmplabel);
  LOG(debug) << "setting data labels incoming of  : " << labels.size() << " with adc of " << adc;
  mADCLabels[adc] = labels;
  LOG(debug) << "setting data labels incoming of  : " << labels.size() << " with adc of " << adc << " now mADCLabels[adc] size is : " << mADCLabels[adc].size();
}

void TrapSimulator::setData(int adc, int it, int data)
{
  //
  // Store ADC data into array of raw data
  // This time enter it element by element.
  //

  if (!checkInitialized())
    return;

  if (adc < 0 || adc >= FeeParam::getNadcMcm()) {
    LOG(error) << "Error: ADC " << adc << " is out of range (0 .. " << FeeParam::getNadcMcm() - 1 << ")";
    return;
  }

  mADCR[adc * mNTimeBin + it] = data << mgkAddDigits;
  mADCF[adc * mNTimeBin + it] = data << mgkAddDigits;
  mADCFilled |= (1 << adc);
}

void TrapSimulator::setBaselines()
{
  //This function exists as in the old simulator, when data was fed into the trapsim, it was done via the whole adc block for the mcm.
  //if the data was zero or out of spec, the baselines were added
  //we now add it by singular ADC, so there are adc channels that never get touched.
  //this fixes that.

  LOG(info) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " for " << mDetector << ":" << mRobPos << ":" << mMcmPos;
  //loop over all adcs.
  for (int adc = 0; adc < FeeParam::getNadcMcm(); adc++) {
    LOG(info) << "Setting baselines for adc: " << adc << " of " << mDetector << ":" << mRobPos << ":" << mMcmPos << hex << mADCFilled << " if of : " << (mADCFilled & (1 << adc));
    if ((mADCFilled & (1 << adc)) == 0) { // adc is empty by construction of mADCFilled.
      LOG(info) << "past if Setting baselines for adc: " << adc << " of " << mDetector << ":" << mRobPos << ":" << mMcmPos;
      for (int timebin = 0; timebin < mNTimeBin; timebin++) {
        mADCR[adc * mNTimeBin + timebin] = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos) + (mgAddBaseline << mgkAddDigits);
        mADCF[adc * mNTimeBin + timebin] = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos) + (mgAddBaseline << mgkAddDigits);
      }
    }
  }
}

void TrapSimulator::setDataPedestal(int adc)
{
  //
  // Store ADC data into array of raw data
  //

  if (!checkInitialized())
    return;

  if (adc < 0 || adc >= FeeParam::getNadcMcm()) {
    return;
  }

  for (int it = 0; it < mNTimeBin; it++) {
    mADCR[adc * mNTimeBin + it] = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos) + (mgAddBaseline << mgkAddDigits);
    mADCF[adc * mNTimeBin + it] = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos) + (mgAddBaseline << mgkAddDigits);
  }
}

bool TrapSimulator::getHit(int index, int& channel, int& timebin, int& qtot, int& ypos, float& y) const
{
  // retrieve the MC hit information (not available in TRAP hardware)

  if (index < 0 || index >= mNHits)
    return false;

  channel = mHits[index].mChannel;
  timebin = mHits[index].mTimebin;
  qtot = mHits[index].mQtot;
  ypos = mHits[index].mYpos;
  y = (float)((((((mRobPos & 0x1) << 2) + (mMcmPos & 0x3)) * 18) << 8) - ((18 * 4 * 2 - 18 * 2 - 1) << 7) -
              (channel << 8) - ypos) *
      (0.635 + 0.03 * (mDetector % 6)) / 256.0;

  return true;
}
//TrapSimualtor::Hit& TrapSimulator::getHit(int index, int& channel, int& timebin, int& qtot, int& ypos, float& y, std::vector<o2::MCCompLabel> &labels) const
//{
//    LOG(fatal) << "for now its not implemented";
//    return nullptr;
//}

int TrapSimulator::getCol(int adc)
{
  //
  // Return column id of the pad for the given ADC channel
  //

  if (!checkInitialized())
    return -1;

  int col = mFeeParam->getPadColFromADC(mRobPos, mMcmPos, adc);
  if (col < 0 || col >= mFeeParam->getNcol())
    return -1;
  else
    return col;
}

int TrapSimulator::packData(std::vector<uint32_t>& buf)
{
  //
  //given the, up to 3 tracklets, pack them according to the define data format.
  //
  //TODO this is left blank so that the dataformats etc. can come in a seperate PR
  //to keep different work seperate.  
  LOG(fatal) << "Called packData of TrapSimualtor ... why? I am going to die, till next pr.";
  return 1;
}

int TrapSimulator::produceRawStream(std::vector<uint32_t>& buf, unsigned int iEv) const
{
  //
  // Produce raw data stream from this MCM and put in buf
  // Returns number of words filled, or negative value
  // with -1 * number of overflowed words
  //
  if (!checkInitialized())
    return 0;

  unsigned int x;
  unsigned int mcmHeader = 0;
  unsigned int adcMask = 0;
  int numberWordsWritten = 0;         // Number numberOverFlowWordsWritten written words
  int numberOverFlowWordsWritten = 0; // Number numberOverFlowWordsWritten overflowed words
  int rawVer = mFeeParam->getRAWversion();
  std::vector<int> adc; //TODO come back an avoid a copy.

  if (!checkInitialized())
    return 0;

  if (mTrapConfig->getTrapReg(TrapConfig::kEBSF, mDetector, mRobPos, mMcmPos) != 0) // store unfiltered data
    adc = mADCR;
  else
    adc = mADCF;

  // Produce ADC mask : nncc cccm mmmm mmmm mmmm mmmm mmmm 1100
  // 				n : unused , c : ADC count, m : selected ADCs
  if (rawVer >= 3 &&
      (mTrapConfig->getTrapReg(TrapConfig::kC15CPUA, mDetector, mRobPos, mMcmPos) & (1 << 13))) { // check for zs flag in TRAP configuration
    int nActiveADC = 0;                                                                           // number numberOverFlowWordsWritten activated ADC bits in a word
    for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
      if (~mZSMap[iAdc] != 0) {       //  0 means not suppressed
        adcMask |= (1 << (iAdc + 4)); // last 4 digit reserved for 1100=0xc
        nActiveADC++;                 // number numberOverFlowWordsWritten 1 in mmm....m
      }
    }

    if ((nActiveADC == 0) &&
        (mTrapConfig->getTrapReg(TrapConfig::kC15CPUA, mDetector, mRobPos, mMcmPos) & (1 << 8))) // check for DEH flag in TRAP configuration
      return 0;

    // assemble adc mask word
    adcMask |= (1 << 30) | ((0x3FFFFFFC) & (~(nActiveADC) << 25)) | 0xC; // nn = 01, ccccc are inverted, 0xc=1100
  }

  // MCM header
  mcmHeader = (1 << 31) | (mRobPos << 28) | (mMcmPos << 24) | ((iEv % 0x100000) << 4) | 0xC;
  if (numberWordsWritten < buf.size())
    buf[numberWordsWritten++] = mcmHeader;
  else
    numberOverFlowWordsWritten++;

  // ADC mask
  if (adcMask != 0) {
    if (numberWordsWritten < buf.size())
      buf[numberWordsWritten++] = adcMask;
    else
      numberOverFlowWordsWritten++;
  }

  // Produce ADC data. 3 timebins are packed into one 32 bits word
  // In this version, different ADC channel will NOT share the same word

  unsigned int aa = 0, a1 = 0, a2 = 0, a3 = 0;

  for (int iAdc = 0; iAdc < 21; iAdc++) {
    if (rawVer >= 3 && ~mZSMap[iAdc] == 0)
      continue; // Zero Suppression, 0 means not suppressed
    aa = !(iAdc & 1) + 2;
    for (int iT = 0; iT < mNTimeBin; iT += 3) {
      a1 = ((iT) < mNTimeBin) ? adc[iAdc * mNTimeBin + iT] >> mgkAddDigits : 0;
      a2 = ((iT + 1) < mNTimeBin) ? adc[iAdc * mNTimeBin + iT + 1] >> mgkAddDigits : 0;
      a3 = ((iT + 2) < mNTimeBin) ? adc[iAdc * mNTimeBin + iT + 2] >> mgkAddDigits : 0;
      x = (a3 << 22) | (a2 << 12) | (a1 << 2) | aa;
      if (numberWordsWritten < buf.size()) {
        buf[numberWordsWritten++] = x;
      } else {
        numberOverFlowWordsWritten++;
      }
    }
  }

  if (numberOverFlowWordsWritten != 0)
    return -numberOverFlowWordsWritten;
  else
    return numberWordsWritten;
}

int TrapSimulator::produceTrackletStream(std::vector<uint32_t>& buf)
{
  //
  // Produce tracklet raw data stream from this MCM and put in buf
  // This is the raw data format for run3 from an MCM
  // Returns number of words filled, or negative value
  // with -1 * number of overflowed words
  //

  if (!checkInitialized())
    return 0;

  int numberWordsWritten = 0; // Number of written words

  // Produce tracklet data.   
  // A MCM Header followed by at most 3 tracklet words.  
  // Although a little confusingly some of the tracklet data is in the header.

  numberWordsWritten = packData(buf);

  return numberWordsWritten;
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

  if (!checkInitialized())
    return;

  // Apply filters sequentially. Bypass is handled by filters
  // since counters and internal registers may be updated even
  // if the filter is bypassed.
  // The first filter takes the data from mADCR and
  // outputs to mADCF.

  //LOG(info) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
  // Non-linearity filter not implemented.
  filterPedestal();
  filterGain();
  filterTail();
  // Crosstalk filter not implemented.
  // LOG(info) << "LEAVE: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
}

void TrapSimulator::filterPedestalInit(int baseline)
{
  // Initializes the pedestal filter assuming that the input has
  // been constant for a long time (compared to the time constant).
  //  LOG(info) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;

  unsigned short fptc = mTrapConfig->getTrapReg(TrapConfig::kFPTC, mDetector, mRobPos, mMcmPos); // 0..3, 0 - fastest, 3 - slowest

  for (int adc = 0; adc < FeeParam::getNadcMcm(); adc++)
    mInternalFilterRegisters[adc].mPedAcc = (baseline << 2) * (1 << mgkFPshifts[fptc]);
  //  LOG(info) << "LEAVE: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
}

unsigned short TrapSimulator::filterPedestalNextSample(int adc, int timebin, unsigned short value)
{
  // Returns the output of the pedestal filter given the input value.
  // The output depends on the internal registers and, thus, the
  // history of the filter.
  ///  LOG(info) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;

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

  // LOG(info) << "LEAVE: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
  if (fpby == 0)
    return value;

  if (inpAdd <= accumulatorShifted)
    return 0;
  else {
    inpAdd = inpAdd - accumulatorShifted;
    if (inpAdd > 0xFFF)
      return 0xFFF;
    else
      return inpAdd;
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
  // LOG(info) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
      int oldadc = mADCF[iAdc * mNTimeBin + iTimeBin];
      mADCF[iAdc * mNTimeBin + iTimeBin] = filterPedestalNextSample(iAdc, iTimeBin, mADCR[iAdc * mNTimeBin + iTimeBin]);
      //    LOG(info) << "mADCF : time : " << iTimeBin << " adc : " << iAdc << " change : " << oldadc << " -> " << mADCF[iAdc * mNTimeBin + iTimeBin];
    }
  }
  // LOG(info) << "BEGIN: " << __FILE__ << ":" << __func__ << ":" << __LINE__ ;
}

void TrapSimulator::filterGainInit()
{
  // Initializes the gain filter. In this case, only threshold
  // counters are reset.

  for (int adc = 0; adc < FeeParam::getNadcMcm(); adc++) {
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
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " with adc = " << adc << " value = " << value;

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
  //if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) << "ENTER: " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " with adc = " << adc << " value = " << value << " Trapconfig values :"  << mgby <<":"<<mgf<<":"<<mga<<":"<<mgta<<":"<<mgtb << ":"<< mgfExtended;
  unsigned int corr; // corrected value

  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) << "after declaring corr adc = " << adc << " value = " << value;
  value &= 0xFFF;
  corr = (value * mgfExtended) >> 11;
  corr = corr > 0xfff ? 0xfff : corr;
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr;
  corr = addUintClipping(corr, mga, 12);
  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr;

  // Update threshold counters
  // not really useful as they are cleared with every new event
  if (!((mInternalFilterRegisters[adc].mGainCounterA == 0x3FFFFFF) || (mInternalFilterRegisters[adc].mGainCounterB == 0x3FFFFFF)))
  // stop when full
  {
    //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr  << " mgtb : " << mgtb;
    if (corr >= mgtb)
      mInternalFilterRegisters[adc].mGainCounterB++;
    else if (corr >= mgta)
      mInternalFilterRegisters[adc].mGainCounterA++;
  }

  //  if(mDetector==75&& mRobPos==5 && mMcmPos==15) LOG(info) <<__LINE__ <<  " adc = " << adc << " value = " << value << " corr  : " << corr  << " mgby : " << mgby;
  //  if (mgby == 1)
  //    return corr;
  //  else
  return value;
}

void TrapSimulator::filterGain()
{
  // Read data from mADCF and apply gain filter.

  for (int adc = 0; adc < FeeParam::getNadcMcm(); adc++) {
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

  if (baseline < 0)
    baseline = mTrapConfig->getTrapReg(TrapConfig::kFPNP, mDetector, mRobPos, mMcmPos);

  ql = lambdaL * (1 - lambdaS) * alphaL;
  qs = lambdaS * (1 - lambdaL) * (1 - alphaL);

  for (int adc = 0; adc < FeeParam::getNadcMcm(); adc++) {
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
  if (inpVolt > aQ)
    aDiff = inpVolt - aQ;
  else
    aDiff = 0;

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
  if (mTrapConfig->getTrapReg(TrapConfig::kFTBY, mDetector, mRobPos, mMcmPos) == 0) // bypass mode, active low
    return value;
  else
    return aDiff;
}

void TrapSimulator::filterTail()
{
  // Apply tail cancellation filter to all data.

  for (int iTimeBin = 0; iTimeBin < mNTimeBin; iTimeBin++) {
    for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
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

  if (!checkInitialized())
    return;

  int eBIS = mTrapConfig->getTrapReg(TrapConfig::kEBIS, mDetector, mRobPos, mMcmPos);
  int eBIT = mTrapConfig->getTrapReg(TrapConfig::kEBIT, mDetector, mRobPos, mMcmPos);
  int eBIL = mTrapConfig->getTrapReg(TrapConfig::kEBIL, mDetector, mRobPos, mMcmPos);
  int eBIN = mTrapConfig->getTrapReg(TrapConfig::kEBIN, mDetector, mRobPos, mMcmPos);

  for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++)
    mZSMap[iAdc] = -1;

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
    iAdc = FeeParam::getNadcMcm() - 1;

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
    for (iAdc = 1; iAdc < FeeParam::getNadcMcm() - 1; iAdc++) {
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

  if (adc > 24) {
    LOG(error) << " adc channel into addHitToFitReg is out of bounds for mFitReg : " << adc;
  }

  if ((timebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos)) &&
      (timebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos))) {
    mFitReg[adc].mQ0 += qtot;
  }

  if ((timebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos)) &&
      (timebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos))) {
    mFitReg[adc].mQ1 += qtot;
  }

  if ((timebin >= mTrapConfig->getTrapReg(TrapConfig::kTPFS, mDetector, mRobPos, mMcmPos)) &&
      (timebin < mTrapConfig->getTrapReg(TrapConfig::kTPFE, mDetector, mRobPos, mMcmPos))) {
    mFitReg[adc].mSumX += timebin;
    mFitReg[adc].mSumX2 += timebin * timebin;
    mFitReg[adc].mNhits++;
    mFitReg[adc].mSumY += ypos;
    mFitReg[adc].mSumY2 += ypos * ypos;
    mFitReg[adc].mSumXY += timebin * ypos;
    //    LOG(debug) << "fitreg[" << adc << "] in timebin " << timebin << ": X=" << mFitReg[adc].mSumX
    //             << ", X2=" << mFitReg[adc].mSumX2 << ", N=" << mFitReg[adc].mNhits << ", Y="
    //           << mFitReg[adc].mSumY << ", Y2=" << mFitReg[adc].mSumY2 << ", XY=" << mFitReg[adc].mSumXY
    //         << ", Q0=" << mFitReg[adc].mQ0 << ", Q1=" << mFitReg[adc].mQ1;
  }

  // register hits (MC info)
  //
  if (mNHits < mgkNHitsMC) {
    mHits[mNHits].mChannel = adc;
    mHits[mNHits].mQtot = qtot;
    mHits[mNHits].mYpos = ypos;
    mHits[mNHits].mTimebin = timebin;
    //  LOG(info) << "in addhit to fit reg labels coming in are of " << labels.size() << " and labels assigned from [mNHits = "<<mNHits<<"] a size of  " << mHits[mNHits].mLabels.size();
    // mHits[mNHits].mLabels = labels;
    //   LOG(info) << "in addhit to fit reg labels coming in are of " << labels.size() << " and labels assigned from [mNHits = "<<mNHits<<"] a size of  " << mHits[mNHits].mLabels.size();
    //TODO link to the labels.
    mNHits++;
    //.emplace_back(adc, timebin, qtot, ypos); // TODO add label indexes into the labels container for all those labels pertaining to this hit.
  } else {
    LOG(warn) << "no space left to store the MC information for the hit >100   mNHits : " << mNHits;
    //print( PRINTRAW| PRINTFILTERED);
  }
  // LOG(info) << "added hit of : "<<  adc<<":"<< qtot<<":"<< ypos<<":"<< timebin; // TODO add label indexes into the labels container for all those labels pertaining to this hit.
}

void TrapSimulator::calcFitreg()
{
  // Preprocessing.
  // Detect the hits and fill the fit registers.
  // Requires 12-bit data from mADCF which means Filter()
  // has to be called before even if all filters are bypassed.
  //??? to be clarified:
  LOG(debug) << "ENTERING : " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " :: " << getDetector() << ":" << getRobPos() << ":" << getMcmPos() << " -------------------- mNHits : " << mNHits;
  unsigned int adcMask = 0xffffffff;

  bool hitQual;
  int adcLeft, adcCentral, adcRight;
  unsigned short timebin, adcch, timebin1, timebin2, qtotTemp;
  short ypos, fromLeft, fromRight, found;
  std::array<unsigned short, 20> qTotal{}; //[19 + 1]; // the last is dummy
  std::array<unsigned short, 6> marked{}, qMarked{};
  unsigned short worse1, worse2;
  int debugstop = 1;

  if (mgStoreClusters) {
    timebin1 = 0;
    timebin2 = mNTimeBin;
  } else {
    // find first timebin to be looked at
    timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPFS, mDetector, mRobPos, mMcmPos);
    if (mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos) < timebin1) {
      timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos);
    }
    if (mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos) < timebin1) {
      timebin1 = mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos);
    }

    // find last timebin to be looked at
    timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPFE, mDetector, mRobPos, mMcmPos);
    if (mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos) > timebin2) {
      timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos);
    }
    if (mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos) > timebin2) {
      timebin2 = mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos);
    }
  }

  // reset the fit registers
  for (auto& fitreg : mFitReg) {
    fitreg.ClearReg();
  }

  //  mFitReg.clear();
  for (int i = 0; i < mNHits; i++) {
    mHits[i].ClearHits(); // do it this way as we dont want to 100 members if we only used 3 ....
  }

  for (timebin = timebin1; timebin < timebin2; timebin++) {
    // first find the hit candidates and store the total cluster charge in qTotal array
    // in case of not hit store 0 there.
    for (adcch = 0; adcch < FeeParam::getNadcMcm() - 2; adcch++) {
      if (((adcMask >> adcch) & 7) == 7) //??? all 3 channels are present in case of ZS
      {
        adcLeft = mADCF[adcch * mNTimeBin + timebin];
        adcCentral = mADCF[(adcch + 1) * mNTimeBin + timebin];
        adcRight = mADCF[(adcch + 2) * mNTimeBin + timebin];

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
        qtotTemp = adcLeft + adcCentral + adcRight;
        /*       if ((qtotTemp > 130)) {
          LOG(info) << "testtree "
                     << "qtot=" << qtotTemp
                     << " qleft=" << adcLeft
                     << " qcent=" << adcCentral
                     << " qright=" << adcRight
                     << " for " << mDetector <<":"<<mRobPos<<":"<<mMcmPos << ":adcleft:"<<adcch<<":t"<<timebin;
        }
*/
        if ((hitQual) &&
            (qtotTemp >= mTrapConfig->getTrapReg(TrapConfig::kTPHT, mDetector, mRobPos, mMcmPos)) &&
            (adcLeft <= adcCentral) &&
            (adcCentral > adcRight)) {
          qTotal[adcch] = qtotTemp;
        } else {
          qTotal[adcch] = 0;
        }
      } else {
        qTotal[adcch] = 0; //jkl
      }
      //     if (qTotal[adcch] != 0)
      //     LOG(info) << "ch " << setw(2) << adcch << "   qTotal " << qTotal[adcch];
    }

    fromLeft = -1;
    adcch = 0;
    found = 0;
    marked[4] = 19; // invalid channel
    marked[5] = 19; // invalid channel
    qTotal[19] = 0;
    int loopcount = 0;
    while ((adcch < 16) && (found < 3)) {
      if (qTotal[adcch] > 0) {
        fromLeft = adcch;
        marked[2 * found + 1] = adcch;
        found++;
      }
      adcch++;
    }

    fromRight = -1;
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
    //    std::cout  <<"Fromleft="<< fromLeft << "  Fromright="<< fromRight << std::endl;
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
      if (marked[4] == marked[5])
        marked[5] = 19;
      for (found = 0; found < 6; found++) {
        qMarked[found] = qTotal[marked[found]] >> 4;
      }

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
        //LOG(debug3) << "Kill ch " << worse1;
      }
      if (worse2 < 19) {
        qTotal[worse2] = 0;
        // LOG(debug3) << "Kill ch " << worse2;
      }
    }

    for (adcch = 0; adcch < 19; adcch++) {
      if (qTotal[adcch] > 0) // the channel is marked for processing
      {
        adcLeft = getDataFiltered(adcch, timebin);
        adcCentral = getDataFiltered(adcch + 1, timebin);
        adcRight = getDataFiltered(adcch + 2, timebin);
        // hit detected, in TRAP we have 4 units and a hit-selection, here we proceed all channels!
        // subtract the pedestal TPFP, clipping instead of wrapping

        int regTPFP = mTrapConfig->getTrapReg(TrapConfig::kTPFP, mDetector, mRobPos, mMcmPos); //TODO put this together with the others as members of trapsim, which is initiliased by det,rob,mcm.
                                                                                               //     LOG(info) << "Hit found, time=" << timebin << ", adcch=" << adcch << "/" << adcch + 1 << "/"
                                                                                               //              << adcch + 2 << ", adc values=" << adcLeft << "/" << adcCentral << "/"
                                                                                               //              << adcRight << ", regTPFP=" << regTPFP << ", TPHT=" << mTrapConfig->getTrapReg(TrapConfig::kTPHT, mDetector, mRobPos, mMcmPos);
        if (adcLeft < regTPFP)
          adcLeft = 0;
        else
          adcLeft -= regTPFP;
        if (adcCentral < regTPFP)
          adcCentral = 0;
        else
          adcCentral -= regTPFP;
        if (adcRight < regTPFP)
          adcRight = 0;
        else
          adcRight -= regTPFP;

        // Calculate the center of gravity
        // checking for adcCentral != 0 (in case of "bad" configuration)
        if (adcCentral == 0)
          continue;
        ypos = 128 * (adcRight - adcLeft) / adcCentral;
        if (ypos < 0)
          ypos = -ypos;
        // make the correction using the position LUT
        ypos = ypos + mTrapConfig->getTrapReg((TrapConfig::TrapReg_t)(TrapConfig::kTPL00 + (ypos & 0x7F)),
                                              mDetector, mRobPos, mMcmPos);
        if (adcLeft > adcRight)
          ypos = -ypos;
        /*   TODO this is left in here as a ref for what was done with labels, its stored externally now figure something out.
*/
        // add the hit to the fitregister
        //      int a=qTotal[adcch] >> mgkAddDigits;
        //      LOG(info) << "calling addHitToFitreg with :" << adcch << " :: " << timebin << " :: " << hex << qTotal[adcch] << dec << " :: shifted bits  :" << 2 << " :: " << ypos;
        //  addHitToFitreg(adcch, timebin, qTotal[adcch] >> 2, ypos);
        LOG(debug) << __func__ << "ADDING HIT FOR in adclabels : " << mADCLabels[adcch].size() << " with adcch of " << adcch << " qtot : " << qTotal[adcch] << " timebin :" << timebin;
        addHitToFitreg(adcch, timebin, qTotal[adcch] >> mgkAddDigits, ypos); //, mADCLabels[adcch]);
        //     LOG(debug) << __FILE__ << ":" << __LINE__ << " :: added hit to fit re ";
      }
    }
  }
  /* 
  for (int iAdc = 0; iAdc < FeeParam::getNadcMcm(); iAdc++) {
    if (mFitReg[iAdc].mNhits != 0) {
      LOG(debug) << "fitreg[" << iAdc << "]: nHits = " << mFitReg[iAdc].mNhits << "]: sumX = " << mFitReg[iAdc].mSumX << ", sumY = " << mFitReg[iAdc].mSumY << ", sumX2 = " << mFitReg[iAdc].mSumX2 << ", sumY2 = " << mFitReg[iAdc].mSumY2 << ", sumXY = " << mFitReg[iAdc].mSumXY;
    }
  }*/
  //print(PRINTRAW);
  LOG(debug) << "LEAVING : " << __FILE__ << ":" << __func__ << ":" << __LINE__ << " :: " << getDetector() << ":" << getRobPos() << ":" << getMcmPos() << " and mNHits : " << mNHits;
}

void TrapSimulator::trackletSelection()
{
  // Select up to 4 tracklet candidates from the fit registers
  // and assign them to the CPUs.
  // LOG(info) << "ENTERING : " << __FILE__ << ":" << __func__ << ":" << __LINE__<<" :: "<< getDetector()<<":"<< getRobPos()<<":"<< getMcmPos();
  unsigned short adcIdx, i, j, ntracks, tmp;
  std::array<unsigned short, 18> trackletCandch{};   // store the adcch[0] and number of hits[1] for all tracklet candidates
  std::array<unsigned short, 18> trackletCandhits{}; // store the adcch[0] and number of hits[1] for all tracklet candidates

  ntracks = 0;
  for (adcIdx = 0; adcIdx < 18; adcIdx++) { // ADCs
    if ((mFitReg[adcIdx].mNhits >= mTrapConfig->getTrapReg(TrapConfig::kTPCL, mDetector, mRobPos, mMcmPos)) &&
        (mFitReg[adcIdx].mNhits + mFitReg[adcIdx + 1].mNhits >= mTrapConfig->getTrapReg(TrapConfig::kTPCT, mDetector, mRobPos, mMcmPos))) {
      trackletCandch[ntracks] = adcIdx;
      trackletCandhits[ntracks] = mFitReg[adcIdx].mNhits + mFitReg[adcIdx + 1].mNhits;
      //   LOG(info) << ntracks << " " << trackletCandch[ntracks] << " " << trackletCandhits[ntracks];
      ntracks++;
    };
  }
  // LOG(info) << "Number of track candidates:"<< ntracks;
  //  for (i = 0; i < ntracks; i++)
  //    LOG(info) << "TRACKS: " << i << " " << trackletCandch[i] << " " << trackletCandhits[i];

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
  for (i = 0; i < ntracks; i++)     // CPUs with tracklets.
    mFitPtr[i] = trackletCandch[i]; // pointer to the left channel with tracklet for CPU[i]
  for (i = ntracks; i < 4; i++)     // CPUs without tracklets
    mFitPtr[i] = 31;                // pointer to the left channel with tracklet for CPU[i] = 31 (invalid)
                                    // LOG(info) << "-------------------------------------------- found " << ntracks << " tracklet candidates";
                                    //  for (i = 0; i < 4; i++)
                                    //   LOG(info) << "fitPtr[" << i << "]: " << mFitPtr[i];

  // reject multiple tracklets
  if (FeeParam::instance()->getRejectMultipleTracklets()) {
    unsigned short counts = 0;
    for (j = 0; j < (ntracks - 1); j++) {
      if (mFitPtr[j] == 31)
        continue;

      for (i = j + 1; i < ntracks; i++) {
        // check if tracklets are from neighbouring ADC channels
        if (TMath::Abs(mFitPtr[j] - mFitPtr[i]) > 1.)
          continue;

        // check which tracklet candidate has higher amount of hits
        if ((mFitReg[mFitPtr[j]].mNhits + mFitReg[mFitPtr[j] + 1].mNhits) >=
            (mFitReg[mFitPtr[i]].mNhits + mFitReg[mFitPtr[i] + 1].mNhits)) {
          mFitPtr[i] = 31;
          counts++;
        } else {
          mFitPtr[j] = 31;
          counts++;
          break;
        }
      }
    }

    ntracks = ntracks - counts;
    //  LOG(debug) << "found " << ntracks << " tracklet candidates";
    //   for (i = 0; i < 4; i++)
    //     LOG(debug) << "fitPtr[" << i << "]: " << mFitPtr[i];
  }
}

void TrapSimulator::fitTracklet()
{
  // Perform the actual tracklet fit based on the fit sums
  // which have been filled in the fit registers.
  //  LOG(debug) << "ENTERING : " << __FILE__ << ":" << __func__ << ":" << __LINE__<<" :: "<< getDetector()<<":"<< getRobPos()<<":"<< getMcmPos() << " and  Tracklet array size is : " << mTrackletArray.size();
  // parameters in fitred.asm (fit program)
  int rndAdd = 0;
  int decPlaces = 5; // must be larger than 1 or change the following code
                     // if (decPlaces >  1)
  rndAdd = (1 << (decPlaces - 1)) + 1;
  // else if (decPlaces == 1)
  //   rndAdd = 1;
  int ndriftDp = 5; // decimal places for drift time
  long shift = ((long)1 << 32);

  // calculated in fitred.asm
  int padrow = ((mRobPos >> 1) << 2) | (mMcmPos >> 2);
  int yoffs = (((((mRobPos & 0x1) << 2) + (mMcmPos & 0x3)) * 18) << 8) -
              ((18 * 4 * 2 - 18 * 2 - 1) << 7);

  // add corrections for mis-alignment
  if (FeeParam::instance()->getUseMisalignCorr()) {
    LOG(debug3) << "using mis-alignment correction";
    yoffs += (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrYcorr, mDetector, mRobPos, mMcmPos);
  }

  yoffs = yoffs << decPlaces; // holds position of ADC channel 1
  int layer = mDetector % 6;
  unsigned int scaleY = (unsigned int)((0.635 + 0.03 * layer) / (256.0 * 160.0e-4) * shift);
  unsigned int scaleD = (unsigned int)((0.635 + 0.03 * layer) / (256.0 * 140.0e-4) * shift);

  int deflCorr = (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCorr, mDetector, mRobPos, mMcmPos);
  int ndrift = (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrNdrift, mDetector, mRobPos, mMcmPos);

  // local variables for calculation
  long mult, temp, denom;       //???
  unsigned int q0, q1, pid;     // charges in the two windows and total charge
  float rawpid, rawz, rawy, rawslope, rawoffset;
  float rawslope4trackletword, rawoffset4trackletword;
  unsigned short nHits;         // number of hits
  int slope, offset;            // slope and offset of the tracklet
  int sumX, sumY, sumXY, sumX2; // fit sums from fit registers
  int sumY2;                    // not used in the current TRAP program, now used for error calculation (simulation only)
  float fitError, fitSlope, fitOffset;
  FitReg *fit0, *fit1; // pointers to relevant fit registers

  //  const uint32_t OneDivN[32] = {  // 2**31/N : exactly like in the TRAP, the simple division here gives the same result!
  //      0x00000000, 0x80000000, 0x40000000, 0x2AAAAAA0, 0x20000000, 0x19999990, 0x15555550, 0x12492490,
  //      0x10000000, 0x0E38E380, 0x0CCCCCC0, 0x0BA2E8B0, 0x0AAAAAA0, 0x09D89D80, 0x09249240, 0x08888880,
  //      0x08000000, 0x07878780, 0x071C71C0, 0x06BCA1A0, 0x06666660, 0x06186180, 0x05D17450, 0x0590B210,
  //      0x05555550, 0x051EB850, 0x04EC4EC0, 0x04BDA120, 0x04924920, 0x0469EE50, 0x04444440, 0x04210840};

  for (int cpu = 0; cpu < 4; cpu++) {
    if (mFitPtr[cpu] == 31) {
      mMCMT[cpu] = 0x10001000; //??? FeeParam::getTrackletEndmarker();
    } else {
      fit0 = &mFitReg[mFitPtr[cpu]];
      fit1 = &mFitReg[mFitPtr[cpu] + 1]; // next channel

      mult = 1;
      mult = mult << (32 + decPlaces);
      mult = -mult;

      // time offset for fit sums
      const int t0 = FeeParam::instance()->getUseTimeOffset() ? (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrTimeOffset, mDetector, mRobPos, mMcmPos) : 0;

      //    LOG(info) << "using time offset t0 = " << t0;

      // Merging
      nHits = fit0->mNhits + fit1->mNhits; // number of hits
      sumX = fit0->mSumX + fit1->mSumX;
      sumX2 = fit0->mSumX2 + fit1->mSumX2;
      denom = ((long)nHits) * ((long)sumX2) - ((long)sumX) * ((long)sumX);

      mult = mult / denom; // exactly like in the TRAP program
      q0 = fit0->mQ0 + fit1->mQ0;
      q1 = fit0->mQ1 + fit1->mQ1;
      sumY = fit0->mSumY + fit1->mSumY + 256 * fit1->mNhits;
      sumXY = fit0->mSumXY + fit1->mSumXY + 256 * fit1->mSumX;
      sumY2 = fit0->mSumY2 + fit1->mSumY2 + 512 * fit1->mSumY + 256 * 256 * fit1->mNhits;

      slope = nHits * sumXY - sumX * sumY;

      //offset  = sumX2*sumY  - sumX*sumXY - t0 * sumX*sumY + t0 * nHits*sumXY;
      offset = sumX2 * sumY - sumX * sumXY;
      offset = offset << 5;
      offset += t0 * nHits * sumXY - t0 * sumX * sumY;
      offset = offset >> 5;

      temp = mult * slope;
      slope = temp >> 32; // take the upper 32 bits
      slope = -slope;
      temp = mult * offset;
      offset = temp >> 32; // take the upper 32 bits

      offset = offset + yoffs;
      //   LOG(info) << "slope = " << slope << ", slope * ndrift = " << slope * ndrift << ", deflCorr: " << deflCorr;
      slope = ((slope * ndrift) >> ndriftDp) + deflCorr;
      offset = offset - (mFitPtr[cpu] << (8 + decPlaces));

      temp = slope;
      temp = temp * scaleD;
      slope = (temp >> 32);
      rawslope4trackletword = temp;
      temp = offset;
      temp = temp * scaleY;
      offset = (temp >> 32);
      rawoffset4trackletword = temp;

      // rounding, like in the TRAP
      slope = (slope + rndAdd) >> decPlaces;
      offset = (offset + rndAdd) >> decPlaces;

      //  LOG(info) << "Det: " << setw(3) << mDetector << ", ROB: " << mRobPos << ", MCM: " << setw(2) << mMcmPos << setw(-1) << ": deflection: " << slope << ", min: " << (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCutStart + 2 * mFitPtr[cpu], mDetector, mRobPos, mMcmPos) << " max : " << (int)mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCutStart + 1 + 2 * mFitPtr[cpu], mDetector, mRobPos, mMcmPos);

      // LOG(info) << "Fit sums: x = " << sumX << ", X = " << sumX2 << ", y = " << sumY << ", Y = " << sumY2 << ", Z = " << sumXY << ", q0 = " << q0 << ", q1 = " << q1;

      fitSlope = (float)(nHits * sumXY - sumX * sumY) / (nHits * sumX2 - sumX * sumX);

      fitOffset = (float)(sumX2 * sumY - sumX * sumXY) / (nHits * sumX2 - sumX * sumX);

      float sx = (float)sumX;
      float sx2 = (float)sumX2;
      float sy = (float)sumY;
      float sy2 = (float)sumY2;
      float sxy = (float)sumXY;
      fitError = sy2 - (sx2 * sy * sy - 2 * sx * sxy * sy + nHits * sxy * sxy) / (nHits * sx2 - sx * sx);
      //fitError = (float) sumY2 - (float) (sumY*sumY) / nHits - fitSlope * ((float) (sumXY - sumX*sumY) / nHits);

      bool rejected = false;
      // deflection range table from DMEM
      if ((slope < ((int)mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCutStart + 2 * mFitPtr[cpu], mDetector, mRobPos, mMcmPos))) ||
          (slope > ((int)mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCutStart + 1 + 2 * mFitPtr[cpu], mDetector, mRobPos, mMcmPos)))) {
        rejected = true;
      }

      //     LOG(info) << "slope : " << slope << " getDmemUnsigned " << mTrapConfig->getDmemUnsigned(mgkDmemAddrDeflCutStart + 2 * mFitPtr[cpu], mDetector, mRobPos, mMcmPos);

      if (rejected && getApplyCut()) {
        mMCMT[cpu] = 0x10001000; //??? FeeParam::getTrackletEndmarker();
      } else {
        if (slope > 63 || slope < -64) { // wrapping in TRAP!
          LOG(info) << "Overflow in slope: " << slope << ", tracklet discarded!";
          mMCMT[cpu] = 0x10001000;
          continue;
        }

        slope = slope & 0x7F; // 7 bit

        if (offset > 0xfff || offset < -0xfff) {
          LOG(warning) << "Overflow in offset";
        }
        offset = offset & 0x1FFF; // 13 bit

        pid = getPID(q0, q1);
        rawpid = pid;

        if (pid > 0xff) {
          LOG(warn) << "Overflow in PID"; // was warn
        }
        pid = pid & 0xFF; // 8 bit, exactly like in the TRAP program

        // assemble and store the tracklet word
        mMCMT[cpu] = (pid << 24) | (padrow << 20) | (slope << 13) | offset;
        //               pid              Z       deflection lenght    Y
        rawz = padrow;
        rawoffset = offset;

        // calculate number of hits and MC label
        //        std::array<int, 3> mcLabel = {-1, -1, -1};
        int nHits0 = 0;
        int nHits1 = 0;

        //        const int maxLabels = 30;
        //        std::array<int, 30> label{}; // up to 30 different labels possible
        //        std::array<int, 30> count{};
        //        int nLabels = 0;
        std::vector<o2::MCCompLabel> localTrackletLabels;
        //we have 21 adc so can fit into a 32bit bit pattern.
        uint32_t adchitbp = 0;
        for (int iHit = 0; iHit < mNHits; iHit++) {
          if ((mHits[iHit].mChannel - mFitPtr[cpu] < 0) ||
              (mHits[iHit].mChannel - mFitPtr[cpu] > 1)) {
            continue;
          }

          // counting contributing hits
          if (mHits[iHit].mTimebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS0, mDetector, mRobPos, mMcmPos) &&
              mHits[iHit].mTimebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE0, mDetector, mRobPos, mMcmPos)) {
            nHits0++;
          }
          if (mHits[iHit].mTimebin >= mTrapConfig->getTrapReg(TrapConfig::kTPQS1, mDetector, mRobPos, mMcmPos) &&
              mHits[iHit].mTimebin < mTrapConfig->getTrapReg(TrapConfig::kTPQE1, mDetector, mRobPos, mMcmPos)) {
            nHits1++;
          }
          LOG(debug) << "setting bit pattern for chanel : " << mHits[iHit].mChannel << " of hit  : " << iHit << std::hex << " bitp before : " << adchitbp << std::dec;
          adchitbp |= (1 << mHits[iHit].mChannel);
          LOG(debug) << "after setting bit pattern for chanel : " << mHits[iHit].mChannel << " of hit  : " << iHit << std::hex << " bitp after : " << adchitbp << std::dec;
          // TODO label calculation only if there is a digitsmanager to get the labels froAm
        }
        bool printoutadcs = false;
        //TODO these debug statemetns are left in to debug the labels later.
        LOG(debug) << "adc bitpatterh is " << std::hex << adchitbp << std::dec;
        for (int i = 0; i < 21; i++) {
          if (adchitbp & (1 << i)) {
            LOG(debug) << "adding labels for adc " << i << "adc bitpatterh is " << std::hex << adchitbp << std::dec << " there are : " << mADCLabels[i].size() << " labels";

            localTrackletLabels.insert(localTrackletLabels.end(), mADCLabels[i].begin(), mADCLabels[i].end()); // append the hit labels to this tracklets temp labels store.
            if (mADCLabels[i].size() == 0) {
              LOG(debug) << "BECAUSE THERE ARE NO LABELS LETS SEE WHICH ADC's DO HAVE LABELS";
              for (int j = 0; j < 21; j++) {
                LOG(debug) << "ADC : " << j << " has " << mADCLabels[j].size() << " labels";
              }
              printoutadcs = true;
            }
          }
        }
        //if(printoutadcs) print(PRINTRAW|PRINTFILTERED);
        //labels come in via adc, now loop over ADC and see which contribute to the hits.

        LOG(debug) << "TrapSim Trackletarray size is : " << mTrackletArray.size() << "  :: adding a track at " << mMCMT[cpu] << ":" << mDetector * 2 + mRobPos % 2 << ":" << mRobPos << ":" << mMcmPos << " LABELS size: " << localTrackletLabels.size();
        mTrackletArray.emplace_back(mDetector * 2 + mRobPos % 2, mRobPos, mMcmPos, rawpid, rawslope, rawoffset, rawslope4trackletword, rawoffset4trackletword);
        int newtrackposition = mTrackletArray.size() - 1;
        //        mTrackletArray[newtrackposition].setLabel(mcLabel);
        mTrackletArray[newtrackposition].setNHits(fit0->mNhits + fit1->mNhits);
        mTrackletArray[newtrackposition].setNHits0(nHits0);
        mTrackletArray[newtrackposition].setNHits1(nHits1);
        mTrackletArray[newtrackposition].setQ0(q0);
        mTrackletArray[newtrackposition].setQ1(q1);
        mTrackletArray[newtrackposition].setSlope(fitSlope);
        mTrackletArray[newtrackposition].setOffset(fitOffset);
        mTrackletArray[newtrackposition].setError(TMath::Sqrt(TMath::Abs(fitError) / nHits));
        LOG(debug) << "adding elements to mTrackletLabels of size " << mTrackletLabels.getIndexedSize() << "::" << mTrackletLabels.getNElements() << " labels for additional labels vector of :" << localTrackletLabels.size() << " labels";
        mTrackletLabels.addElements(mTrackletLabels.getIndexedSize(), localTrackletLabels);
        LOG(debug) << "elements in mTrackletLabels is size " << mTrackletLabels.getIndexedSize() << "::" << mTrackletLabels.getNElements() << " labels for additional labels vector of :" << localTrackletLabels.size() << " labels";
        // store cluster information (if requested)
        if (mgStoreClusters) {
          std::vector<float> res(getNumberOfTimeBins());
          std::vector<float> qtot(getNumberOfTimeBins());
          for (int iTimebin = 0; iTimebin < getNumberOfTimeBins(); ++iTimebin) {
            res[iTimebin] = 0;
            qtot[iTimebin] = 0;
          }

          for (int iHit = 0; iHit < mNHits; iHit++) {
            int timebin = mHits[iHit].mTimebin;

            // check if hit contributes
            if (mHits[iHit].mChannel == mFitPtr[cpu]) {
              //             for (int i=0;i<mNHits;i++) LOG(info) << "mHits.["<<i << "].mTimebin="<< mHits[i].mTimebin;
              res[timebin] = mHits[iHit].mYpos - (fitSlope * timebin + fitOffset);
              qtot[timebin] = mHits[iHit].mQtot;
            } else if (mHits[iHit].mChannel == mFitPtr[cpu] + 1) {
              res[timebin] = mHits[iHit].mYpos + 256 - (fitSlope * timebin + fitOffset);
              qtot[timebin] = mHits[iHit].mQtot;
            }
          }
          mTrackletArray[newtrackposition].setClusters(res, qtot, getNumberOfTimeBins());
        }

        //TODO dont leave this commented out !
        if (fitError < 0) {
          LOG(warn) << "fit slope: " << fitSlope << ", offset: " << fitOffset << ", error: " << TMath::Sqrt(TMath::Abs(fitError) / nHits);
          LOG(info) << "Strange fit error: " << fitError << " from Sx: " << sumX << ", Sy: " << sumY << ", Sxy: " << sumXY << ", Sx2: " << sumX2 << ", Sy2: " << sumY2 << ", nHits: " << nHits;
        }
      }
    }
  }
  //  LOG(info) << "LEAVING : " << __FILE__ << ":" << __func__ << ":" << __LINE__<<" :: "<< getDetector()<<":"<< getRobPos()<<":"<< getMcmPos() << " and  Tracklet array size is : " << mTrackletArray.size();
}

void TrapSimulator::tracklet()
{
  // Run the tracklet calculation by calling sequentially:
  // CalcFitreg(); TrackletSelection(); FitTracklet()
  // and store the tracklets

  if (!mInitialized) {
    return;
  }
  mTrackletArray.clear();

  calcFitreg();
  if (mNHits == 0) {
    return;
  }
  trackletSelection();
  fitTracklet();
}

void TrapSimulator::getTracklets(std::vector<Tracklet>& TrackletStore)
{
  // simply returns the found tracklets for the O2 dpl to then do its thing.
  //
  TrackletStore.insert(std::end(TrackletStore), std::begin(mTrackletArray), std::end(mTrackletArray));
  //std::copy(mTrackletArray.begin(),mTrackletArray.end(),std::back_inserter(TrackletStore));
}

// ******************************
// PID section
//
// Memory area for the LUT: 0xC100 to 0xC3FF
//
// The addresses for the parameters (the order is optimized for maximum calculation speed in the MCMs):
// 0xC028: cor1
// 0xC029: nBins(sF)
// 0xC02A: cor0
// 0xC02B: TableLength
// Defined in TrapConfig.h
//
// The algorithm implemented in the TRAP program of the MCMs (Venelin Angelov)
//  1) set the read pointer to the beginning of the Parameters in DMEM
//  2) shift right the FitReg with the Q0 + (Q1 << 16) to get Q1
//  3) read cor1 with rpointer++
//  4) start cor1*Q1
//  5) read nBins with rpointer++
//  6) start nBins*cor1*Q1
//  7) read cor0 with rpointer++
//  8) swap hi-low parts in FitReg, now is Q1 + (Q0 << 16)
//  9) shift right to get Q0
// 10) start cor0*Q0
// 11) read TableLength
// 12) compare cor0*Q0 with nBins
// 13) if >=, clip cor0*Q0 to nBins-1
// 14) add cor0*Q0 to nBins*cor1*Q1
// 15) compare the result with TableLength
// 16) if >=, clip to TableLength-1
// 17) read from the LUT 8 bits
// TODO the new program from Venelin will be different.

int TrapSimulator::getPID(int q0, int q1)
{
  // return PID calculated from charges accumulated in two time windows

  unsigned long long addrQ0;
  unsigned long long addr;

  unsigned int nBinsQ0 = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTnbins, mDetector, mRobPos, mMcmPos); // number of bins in q0 / 4 !!
  unsigned int pidTotalSize = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTLength, mDetector, mRobPos, mMcmPos);
  if (nBinsQ0 == 0 || pidTotalSize == 0) // make sure we don't run into trouble if the value for Q0 is not configured
    return 0;                            // Q1 not configured is ok for 1D LUT

  unsigned long corrQ0 = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTcor0, mDetector, mRobPos, mMcmPos);
  unsigned long corrQ1 = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTcor1, mDetector, mRobPos, mMcmPos);
  if (corrQ0 == 0) // make sure we don't run into trouble if one of the values is not configured
    return 0;

  addrQ0 = corrQ0;
  addrQ0 = (((addrQ0 * q0) >> 16) >> 16); // because addrQ0 = (q0 * corrQ0) >> 32; does not work for unknown reasons

  if (addrQ0 >= nBinsQ0) { // check for overflow
                           //    LOG(debug3) << "Overflow in q0: " << addrQ0 << "/4 is bigger then " << nBinsQ0;
    addrQ0 = nBinsQ0 - 1;
  }

  addr = corrQ1;
  addr = (((addr * q1) >> 16) >> 16);
  addr = addrQ0 + nBinsQ0 * addr; // because addr = addrQ0 + nBinsQ0* (((corrQ1*q1)>>32); does not work

  if (addr >= pidTotalSize) {
    //  LOG(debug3) << "Overflow in q1. Address " << addr << "/4 is bigger then " << pidTotalSize;
    addr = pidTotalSize - 1;
  }

  // For a LUT with 11 input and 8 output bits, the first memory address is set to  LUT[0] | (LUT[1] << 8) | (LUT[2] << 16) | (LUT[3] << 24)
  // and so on
  unsigned int result = mTrapConfig->getDmemUnsigned(mgkDmemAddrLUTStart + (addr / 4), mDetector, mRobPos, mMcmPos);
  return (result >> ((addr % 4) * 8)) & 0xFF;
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
    if (sum > maxv)
      sum = maxv;
  } else {
    if ((sum < a) || (sum < b))
      sum = 0xFFFFFFFF;
  }
  return sum;
}

void TrapSimulator::sort2(unsigned short idx1i, unsigned short idx2i,
                          unsigned short val1i, unsigned short val2i,
                          unsigned short* const idx1o, unsigned short* const idx2o,
                          unsigned short* const val1o, unsigned short* const val2o) const
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

void TrapSimulator::sort3(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i,
                          unsigned short val1i, unsigned short val2i, unsigned short val3i,
                          unsigned short* const idx1o, unsigned short* const idx2o, unsigned short* const idx3o,
                          unsigned short* const val1o, unsigned short* const val2o, unsigned short* const val3o)
{
  // sorting for tracklet selection

  int sel;

  if (val1i > val2i)
    sel = 4;
  else
    sel = 0;
  if (val2i > val3i)
    sel = sel + 2;
  if (val3i > val1i)
    sel = sel + 1;
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

void TrapSimulator::sort6To4(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i, unsigned short idx4i, unsigned short idx5i, unsigned short idx6i,
                             unsigned short val1i, unsigned short val2i, unsigned short val3i, unsigned short val4i, unsigned short val5i, unsigned short val6i,
                             unsigned short* const idx1o, unsigned short* const idx2o, unsigned short* const idx3o, unsigned short* const idx4o,
                             unsigned short* const val1o, unsigned short* const val2o, unsigned short* const val3o, unsigned short* const val4o)
{
  // sorting for tracklet selection

  unsigned short idx21s, idx22s, idx23s, dummy;
  unsigned short val21s, val22s, val23s;
  unsigned short idx23as, idx23bs;
  unsigned short val23as, val23bs;

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

void TrapSimulator::sort6To2Worst(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i, unsigned short idx4i, unsigned short idx5i, unsigned short idx6i,
                                  unsigned short val1i, unsigned short val2i, unsigned short val3i, unsigned short val4i, unsigned short val5i, unsigned short val6i,
                                  unsigned short* const idx5o, unsigned short* const idx6o)
{
  // sorting for tracklet selection

  unsigned short idx21s, idx22s, idx23s, dummy1, dummy2, dummy3, dummy4, dummy5;
  unsigned short val21s, val22s, val23s;
  unsigned short idx23as, idx23bs;
  unsigned short val23as, val23bs;

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

      LOG(debug3) << "read: 0x" << hex << header;

      if (header & 0x01) // single data
      {
        dat = (header >> 2) & 0xFFFF;    // 16 bit data
        caddr = (header >> 18) & 0x3FFF; // 14 bit address

        if (caddr != 0x1FFF) // temp!!! because the end marker was wrong
        {
          if (header & 0x02) // check if > 16 bits
          {
            dataHi = *data;
            LOG(debug3) << "read: 0x" << hex << dataHi;
            data++;
            idx++;
            err += ((dataHi ^ (dat | 1)) & 0xFFFF) != 0;
            dat = (dataHi & 0xFFFF0000) | dat;
          }
          LOG(debug3) << "addr=0x" << hex << caddr << "(" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ") data=0x" << hex << dat;
          if (!cfg->poke(caddr, dat, det, rob, mcm)) {
            LOG(debug3) << "(single-write): non-existing address 0x" << std::hex << caddr << " containing 0x" << std::hex << header;
          }
          if (idx > size) {
            LOG(debug3) << "(single-write): no more data, missing end marker";
            return -err;
          }
        } else {
          LOG(debug3) << "(single-write): address 0x" << setw(4) << std::hex << caddr << " => old endmarker?" << std::dec;
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
                LOG(debug3) << "read 0x" << setw(8) << std::hex << header << std::dec;
                data++;
                idx++;
                err += (header & 1);
                header = header >> 1;
                bitcnt = 31 - bwidth;
              }
              LOG(debug3) << "addr=0x" << setw(4) << std::hex << caddr << "(" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ") data=0x" << setw(8) << std::hex << (header & msk);
              if (!cfg->poke(caddr, header & msk, det, rob, mcm)) {
                LOG(debug3) << "(single-write): non-existing address 0x" << setw(4) << std::hex << caddr << " containing 0x" << setw(8) << std::hex << header << std::dec;
              }

              caddr += step;
              header = header >> bwidth;
              if (idx >= size) {
                LOG(debug3) << "(block-write): no end marker! " << idx << " words read";
                return -err;
              }
            }
            break;
          } // end case 5-15
          case 31: {
            while (nwords > 0) {
              header = *data;
              LOG(debug3) << "read 0x" << setw(8) << std::hex << header;
              data++;
              idx++;
              nwords--;
              err += (header & 1);

              LOG(debug3) << "addr=0x" << hex << setw(4) << caddr << " (" << cfg->getRegName(cfg->getRegByAddress(caddr)) << ")  data=0x" << hex << setw(8) << (header >> 1);
              if (!cfg->poke(caddr, header >> 1, det, rob, mcm)) {
                LOG(debug3) << "(single-write): non-existing address 0x" << setw(4) << std::hex << " containing 0x" << setw(8) << std::hex << header << std::dec;
              }

              caddr += step;
              if (idx >= size) {
                LOG(debug3) << "no end marker! " << idx << " words read";
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

