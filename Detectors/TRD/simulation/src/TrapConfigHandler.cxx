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

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  Trap configuraton handler                                             //
//                                                                        //
//  based on work by U. Westerhoff                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <iomanip>

#include "TRDBase/FeeParam.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/CalOnlineGainTables.h"
#include "TRDBase/PadResponse.h"
#include "TRDSimulation/TrapConfigHandler.h"
#include "TRDSimulation/TrapConfig.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/Constants.h"
#include "TMath.h"
#include "TGeoMatrix.h"
#include "TGraph.h"

using namespace std;
using namespace o2::trd;
using namespace o2::trd::constants;

TrapConfigHandler::TrapConfigHandler(TrapConfig* cfg) : mFeeParam(), mRestrictiveMask((0x3ffff << 11) | (0x1f << 6) | 0x3f), mTrapConfig(cfg), mGtbl()
{
}

TrapConfigHandler::~TrapConfigHandler() = default;

void TrapConfigHandler::init()
{
  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  // setup of register allocation
  // I/O configuration which we don't care about
  mTrapConfig->setTrapRegAlloc(TrapConfig::kSEBDOU, TrapConfig::kAllocNone);
  // position look-up table by layer
  for (int iBin = 0; iBin < 128; iBin++) {
    mTrapConfig->setTrapRegAlloc((TrapConfig::TrapReg_t)(TrapConfig::kTPL00 + iBin), TrapConfig::kAllocByLayer);
  }
  // ... individual
  mTrapConfig->setTrapRegAlloc(TrapConfig::kC14CPUA, TrapConfig::kAllocByMCM);
  mTrapConfig->setTrapRegAlloc(TrapConfig::kC15CPUA, TrapConfig::kAllocByMCM);

  // setup of DMEM allocation
  for (int iAddr = TrapConfig::mgkDmemStartAddress;
       iAddr < (TrapConfig::mgkDmemWords + TrapConfig::mgkDmemStartAddress); iAddr++) {

    if (iAddr == TrapSimulator::mgkDmemAddrDeflCorr) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByMCMinSM);

    } else if (iAddr == TrapSimulator::mgkDmemAddrNdrift) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByDetector);

    } else if ((iAddr >= TrapSimulator::mgkDmemAddrDeflCutStart) && (iAddr <= TrapSimulator::mgkDmemAddrDeflCutEnd)) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByMCMinSM);

    } else if ((iAddr >= TrapSimulator::mgkDmemAddrTrackletStart) && (iAddr <= TrapSimulator::mgkDmemAddrTrackletEnd)) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByMCM);

    } else if ((iAddr >= TrapSimulator::mgkDmemAddrLUTStart) && (iAddr <= TrapSimulator::mgkDmemAddrLUTEnd)) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocGlobal);

    } else if (iAddr == TrapSimulator::mgkDmemAddrLUTcor0) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByMCMinSM);

    } else if (iAddr == TrapSimulator::mgkDmemAddrLUTcor1) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocByMCMinSM);

    } else if (iAddr == TrapSimulator::mgkDmemAddrLUTnbins) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocGlobal);

    } else if (iAddr == TrapSimulator::mgkDmemAddrLUTLength) {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocGlobal);

    } else {
      mTrapConfig->setDmemAlloc(iAddr, TrapConfig::kAllocGlobal);
    }
  }
  mFeeParam = FeeParam::instance();
}

void TrapConfigHandler::resetMCMs()
{
  //
  // Reset all MCM registers and DMEM
  //

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  mTrapConfig->resetRegs();
  mTrapConfig->resetDmem();
}

int TrapConfigHandler::loadConfig()
{
  // load a default configuration which is suitable for simulation
  // for a detailed description of the registers see the TRAP manual
  // if you want to resimulate tracklets on real data use the appropriate config instead

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return -1;
  }

  // prepare mFeeParam
  // ndrift (+ 5 binary digits)
  mFeeParam->setNtimebins(20 << 5);
  // deflection + tilt correction
  mFeeParam->setRawOmegaTau(0.16133);
  // deflection range table
  mFeeParam->setRawPtMin(0.1);
  // magnetic field
  mFeeParam->setRawMagField(0.0);
  // scaling factors for q0, q1
  mFeeParam->setRawScaleQ0(0);
  mFeeParam->setRawScaleQ1(0);
  // disable length correction and tilting correction
  mFeeParam->setRawLengthCorrectionEnable(false);
  mFeeParam->setRawTiltCorrectionEnable(false);

  for (int iDet = 0; iDet < 540; iDet++) {
    // HC header configuration bits
    mTrapConfig->setTrapReg(TrapConfig::kC15CPUA, 0x2102, iDet); // zs, deh

    // no. of timebins
    mTrapConfig->setTrapReg(TrapConfig::kC13CPUA, 24, iDet);

    // pedestal filter
    mTrapConfig->setTrapReg(TrapConfig::kFPNP, 4 * 10, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFPTC, 0, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFPBY, 0, iDet); // bypassed!

    // gain filter
    for (int adc = 0; adc < 20; adc++) {
      mTrapConfig->setTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGA0 + adc), 40, iDet);
      mTrapConfig->setTrapReg(TrapConfig::TrapReg_t(TrapConfig::kFGF0 + adc), 15, iDet);
    }
    mTrapConfig->setTrapReg(TrapConfig::kFGTA, 20, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFGTB, 2060, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFGBY, 0, iDet); // bypassed!

    // tail cancellation
    mTrapConfig->setTrapReg(TrapConfig::kFTAL, 200, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFTLL, 0, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFTLS, 200, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kFTBY, 0, iDet);

    // tracklet calculation
    mTrapConfig->setTrapReg(TrapConfig::kTPQS0, 5, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPQE0, 10, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPQS1, 11, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPQE1, 20, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPFS, 5, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPFE, 20, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPVBY, 0, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPVT, 10, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPHT, 150, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPFP, 40, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPCL, 1, iDet);
    mTrapConfig->setTrapReg(TrapConfig::kTPCT, 10, iDet);

    // apply mFeeParams
    configureDyCorr(iDet);
    configureDRange(iDet);    // deflection range
    configureNTimebins(iDet); // timebins in the drift region
    configurePIDcorr(iDet);   // scaling parameters for the PID

    // event buffer
    mTrapConfig->setTrapReg(TrapConfig::kEBSF, 1, iDet); // 0: store filtered; 1: store unfiltered

    // zs applied to data stored in event buffer (sel. by EBSF)
    mTrapConfig->setTrapReg(TrapConfig::kEBIS, 15, iDet);   // single indicator threshold (plus two digits)
    mTrapConfig->setTrapReg(TrapConfig::kEBIT, 30, iDet);   // sum indicator threshold (plus two digits)
    mTrapConfig->setTrapReg(TrapConfig::kEBIL, 0xf0, iDet); // lookup table
    mTrapConfig->setTrapReg(TrapConfig::kEBIN, 0, iDet);    // neighbour sensitivity

    // raw data
    mTrapConfig->setTrapReg(TrapConfig::kNES, (0x0000 << 16) | 0x1000, iDet);
  }

  // ****** hit position LUT

  // now calculate it from PRF
  PadResponse padresponse;
  double padResponse[3];  // pad response left, central, right
  double padResponseR[3]; // pad response left, central, right
  double padResponseL[3]; // pad response left, central, right

  for (int iLayer = 0; iLayer < 6; iLayer++) {
    TGraph gr(128);
    for (int iBin = 0; iBin < 256 * 0.5; iBin++) {
      padresponse.getPRF(1., iBin * 1. / 256., iLayer, padResponse);
      padresponse.getPRF(1., iBin * 1. / 256. - 1., iLayer, padResponseR);
      padresponse.getPRF(1., iBin * 1. / 256. + 1., iLayer, padResponseL);
      gr.SetPoint(iBin, (0.5 * (padResponseR[1] - padResponseL[1]) / padResponse[1] * 256), iBin);
    }
    for (int iBin = 0; iBin < 128; iBin++) {
      int corr = (int)(gr.Eval(iBin)) - iBin;
      if (corr < 0) {
        corr = 0;
      } else if (corr > 31) {
        corr = 31;
      }
      for (int iStack = 0; iStack < 540 / 6; iStack++) {
        mTrapConfig->setTrapReg((TrapConfig::TrapReg_t)(TrapConfig::kTPL00 + iBin), corr, 6 * iStack + iLayer);
      }
    }
  }
  // ****** hit position LUT configuration end

  return 0;
}

int TrapConfigHandler::loadConfig(std::string filename)
{
  //
  // load a TRAP configuration from a file
  // The file format is the format created by the standalone
  // command coder scc / show_cfdat but without the running number
  // scc /show_cfdat adds as the first column
  // which are two tools to inspect/export configurations from wingDB
  //

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return -1;
  }

  int ignoredLines = 0;
  int ignoredCmds = 0;
  int readLines = 0;

  LOG(debug3) << "Processing file " << filename.c_str();
  std::ifstream infile;
  infile.open(filename.c_str(), std::ifstream::in);
  if (!infile.is_open()) {
    LOG(error) << "Can not open MCM configuration file  " << filename;
    return false;
  }

  int addr;

  // reset restrictive mask
  mRestrictiveMask = (0x3ffff << 11) | (0x1f << 6) | 0x3f;
  char linebuffer[512];
  istringstream line;

  while (infile.getline(linebuffer, 512) && infile.good()) {
    unsigned int cmd = 999;
    int data = -1;
    int extali = -1;
    line.clear();
    line.str(linebuffer);
    data = -1;
    line >> std::skipws >> cmd >> addr >> data >> extali; // the lines read from config file can contain additional columns.
    // Therefore the detour via istringstream

    if (cmd != 999 && addr != -1 && extali != -1) {

      if (cmd == mgkScsnCmdWrite) {
        for (int det = 0; det < MAXCHAMBER; det++) {
          unsigned int rocpos = (1 << (Geometry::getSector(det) + 11)) | (1 << (Geometry::getStack(det) + 6)) | (1 << Geometry::getLayer(det));
          LOG(debug) << "checking restriction: mask=0x" << hex << std::setw(8) << mRestrictiveMask << " rocpos=0x" << hex << std::setw(8) << rocpos;
          if ((mRestrictiveMask & rocpos) == rocpos) {
            LOG(debug) << "match:"
                       << " " << cmd << " " << extali << " " << addr << " " << data;
            addValues(det, cmd, extali, addr, data);
          }
        }
      }

      else if (cmd == mgkScsnLTUparam) {
        //     processLTUparam(extali, addr, data);
      }

      else if (cmd == mgkScsnCmdRestr) {
        mRestrictiveMask = data;
        LOG(debug) << "updated restrictive mask to 0x" << hex << std::setw(8) << mRestrictiveMask;
      }

      else if ((cmd == mgkScsnCmdReset) ||
               (cmd == mgkScsnCmdRobReset)) {
        mTrapConfig->resetRegs();
      }

      else if (cmd == mgkScsnCmdSetHC) {
        int fullVersion = ((data & 0x7F00) >> 1) | (data & 0x7f);

        for (int iDet = 0; iDet < MAXCHAMBER; iDet++) {
          int smls = (Geometry::getSector(iDet) << 6) | (Geometry::getLayer(iDet) << 3) | Geometry::getStack(iDet);

          for (int iRob = 0; iRob < 8; iRob++) {
            // HC mergers
            mTrapConfig->setTrapReg(TrapConfig::kC14CPUA, 0xc << 16, iDet, iRob, 17);
            mTrapConfig->setTrapReg(TrapConfig::kC15CPUA, ((1 << 29) | (fullVersion << 15) | (1 << 12) | (smls << 1) | (iRob % 2)), iDet, iRob, 17);

            // board mergers
            mTrapConfig->setTrapReg(TrapConfig::kC14CPUA, 0, iDet, iRob, 16);
            mTrapConfig->setTrapReg(TrapConfig::kC15CPUA, ((1 << 29) | (fullVersion << 15) | (1 << 12) | (smls << 1) | (iRob % 2)), iDet, iRob, 16);

            // and now for the others
            for (int iMcm = 0; iMcm < 16; iMcm++) {
              mTrapConfig->setTrapReg(TrapConfig::kC14CPUA, iMcm | (iRob << 4) | (3 << 16), iDet, iRob, iMcm);
              mTrapConfig->setTrapReg(TrapConfig::kC15CPUA, ((1 << 29) | (fullVersion << 15) | (1 << 12) | (smls << 1) | (iRob % 2)), iDet, iRob, iMcm);
            }
          }
        }
      }

      else if ((cmd == mgkScsnCmdRead) ||
               (cmd == mgkScsnCmdPause) ||
               (cmd == mgkScsnCmdPtrg) ||
               (cmd == mgkScsnCmdHwPtrg) ||
               (cmd == mgkScsnCmdRobPower) ||
               (cmd == mgkScsnCmdTtcRx) ||
               (cmd == mgkScsnCmdMcmTemp) ||
               (cmd == mgkScsnCmdOri) ||
               (cmd == mgkScsnCmdPM)) {
        LOG(debug2) << "ignored SCSN command: " << cmd << " " << addr << " " << data << " " << extali;
      }

      else {
        LOG(warn) << "unknown SCSN command: " << cmd << " " << addr << " " << data << " " << extali;
        ignoredCmds++;
      }

      readLines++;
    }

    else if (!infile.eof() && !infile.good()) {
      infile.clear();
      infile.ignore(256, '\n');
      ignoredLines++;
    }

    if (!infile.eof()) {
      infile.clear();
    }
  }

  infile.close();

  LOG(debug3) << "Ignored lines: " << ignoredLines << ", ignored cmds: " << ignoredCmds;

  if (ignoredLines > readLines) {
    LOG(error) << "More than 50% of the input file could not be processed. Perhaps you should check the input file " << filename.c_str();
  }

  return true;
}

int TrapConfigHandler::setGaintable(CalOnlineGainTables& gtbl)
{
  mGtbl = gtbl;
  return 0;
}

void TrapConfigHandler::processLTUparam(int dest, int addr, unsigned int data)
{
  //
  // Process the LTU parameters and stores them in internal class variables
  // or transfer the stored values to TrapConfig, depending on the dest parameter
  //

  switch (dest) {

    case 0: // set the parameters in TrapConfig
      for (int det = 0; det < MAXCHAMBER; det++) {
        configureDyCorr(det);
        configureDRange(det);    // deflection range
        configureNTimebins(det); // timebins in the drift region
        configurePIDcorr(det);   // scaling parameters for the PID
      }
      break;

    case 1: // set variables
      switch (addr) {

        case 0:
          mFeeParam->setPtMin(data);
          break; // pt_min in GeV/c (*1000)
        case 1:
          mFeeParam->setMagField(data);
          break; // B in T (*1000)
        case 2:
          mFeeParam->setOmegaTau(data);
          break; // omega*tau
        case 3:
          mFeeParam->setNtimebins(data);
          break;
          // ntimbins: drift time (for 3 cm) in timebins (5 add. bin. digits)
        case 4:
          mFeeParam->setScaleQ0(data);
          break;
        case 5:
          mFeeParam->setScaleQ1(data);
          break;
        case 6:
          mFeeParam->setLengthCorrectionEnable(data);
          break;
        case 7:
          mFeeParam->setTiltCorrectionEnable(data);
          break;
      }
      break;

    default:
      LOG(error) << "dest " << dest << " not implemented";
  }
}

void TrapConfigHandler::configureNTimebins(int det)
{
  //
  // set timebins in the drift region
  //

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  addValues(det, mgkScsnCmdWrite, 127, TrapSimulator::mgkDmemAddrNdrift, mFeeParam->getNtimebins());
}

void TrapConfigHandler::configureDyCorr(int det)
{
  //
  //  Deflection length correction
  //  due to Lorentz angle and tilted pad correction
  //  This correction is in units of padwidth / (256*32)
  //

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  int nRobs = Geometry::getStack(det) == 2 ? 6 : 8;

  for (int r = 0; r < nRobs; r++) {
    for (int m = 0; m < 16; m++) {
      int dest = 1 << 10 | r << 7 | m;
      int dyCorrInt = mFeeParam->getDyCorrection(det, r, m);
      addValues(det, mgkScsnCmdWrite, dest, TrapSimulator::mgkDmemAddrDeflCorr, dyCorrInt);
    }
  }
}

void TrapConfigHandler::configureDRange(int det)
{
  //
  // deflection range LUT
  // range calculated according to B-field (in T) and pt_min (in GeV/c)
  // if pt_min < 0.1 GeV/c the maximal allowed range for the tracklet
  // deflection (-64..63) is used
  //
  // TODO might need to be updated depending in the FEE configuration for Run 3

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  int nRobs = Geometry::getStack(det) == 2 ? 6 : 8;

  int dyMinInt;
  int dyMaxInt;

  for (int r = 0; r < nRobs; r++) {
    for (int m = 0; m < 16; m++) {
      for (int c = 0; c < 18; c++) {

        // cout << "maxdefl: " << maxDeflAngle << ", localPhi " << localPhi << endl;
        // cout << "r " << r << ", m" << m << ", c " << c << ", min angle: " << localPhi-maxDeflAngle << ", max: " << localPhi+maxDeflAngle
        //<< ", min int: " << dyMinInt << ", max int: " << dyMaxInt << endl;
        int dest = 1 << 10 | r << 7 | m;
        int lutAddr = TrapSimulator::mgkDmemAddrDeflCutStart + 2 * c;
        mFeeParam->getDyRange(det, r, m, c, dyMinInt, dyMaxInt);
        addValues(det, mgkScsnCmdWrite, dest, lutAddr + 0, dyMinInt);
        addValues(det, mgkScsnCmdWrite, dest, lutAddr + 1, dyMaxInt);
      }
    }
  }
}

void TrapConfigHandler::printGeoTest()
{
  //
  // Prints some information about the geometry. Only for debugging
  //

  int sm = 0;
  //   for(int sm=0; sm<6; sm++) {
  for (int stack = 0; stack < 5; stack++) {
    for (int layer = 0; layer < 6; layer++) {

      int det = sm * 30 + stack * 6 + layer;
      for (int r = 0; r < 6; r++) {
        for (int m = 0; m < 16; m++) {
          for (int c = 7; c < 8; c++) {
            cout << stack << ";" << layer << ";" << r << ";" << m
                 << ";" << mFeeParam->getX(det, r, m)
                 << ";" << mFeeParam->getLocalY(det, r, m, c)
                 << ";" << mFeeParam->getLocalZ(det, r, m) << endl;
          }
        }
      }
    }
  }
  // }
}

void TrapConfigHandler::configurePIDcorr(int det)
{
  //
  // Calculate the MCM individual correction factors for the PID
  // and transfer them to TrapConfig
  //

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return;
  }

  static const int addrLUTcor0 = TrapSimulator::mgkDmemAddrLUTcor0;
  static const int addrLUTcor1 = TrapSimulator::mgkDmemAddrLUTcor1;

  unsigned int cor0;
  unsigned int cor1;
  int readoutboard;
  int mcm;
  // int nRobs = Geometry::getStack(det) == 2 ? 6 : 8;
  int MaxRows;
  FeeParam* feeparam = FeeParam::instance();
  if (Geometry::getStack(det) == 2) {
    MaxRows = NROWC0;
  } else {
    MaxRows = NROWC1;
  }
  int MaxCols = NCOLUMN;
  for (int row = 0; row < MaxRows; row++) { //TODO put this back to rob/mcm and not row/col as done in TrapSimulator
    for (int col = 0; col < MaxCols; col++) {
      readoutboard = feeparam->getROBfromPad(row, col);
      mcm = feeparam->getMCMfromPad(row, col);
      int dest = 1 << 10 | readoutboard << 7 | mcm;
      //TODO impelment a method for determining if gaintables are valid, used to be if pointer was valid.
      if (mGtbl.getMCMGain(det, row, col) != 0.0) { //TODO check this logic there might be problems here.
        mFeeParam->getCorrectionFactors(det, readoutboard, mcm, 9, cor0, cor1, mGtbl.getMCMGain(det, row, col));
      } else {
        mFeeParam->getCorrectionFactors(det, readoutboard, mcm, 9, cor0, cor1);
      }
      addValues(det, mgkScsnCmdWrite, dest, addrLUTcor0, cor0);
      addValues(det, mgkScsnCmdWrite, dest, addrLUTcor1, cor1);
    }
  }
}

bool TrapConfigHandler::addValues(unsigned int det, unsigned int cmd, unsigned int extali, int addr, unsigned int data)
{
  // transfer the informations provided by LoadConfig to the internal class variables

  if (!mTrapConfig) {
    LOG(error) << "No TRAPconfig given";
    return false;
  }

  if (cmd != mgkScsnCmdWrite) {
    LOG(error) << "Invalid command received: " << cmd;
    return false;
  }

  TrapConfig::TrapReg_t mcmReg = mTrapConfig->getRegByAddress(addr);
  int rocType = Geometry::getStack(det) == 2 ? 0 : 1;

  static const int mcmListSize = 40; // 40 is more or less arbitrary
  int mcmList[mcmListSize];

  // configuration registers
  if (mcmReg >= 0 && mcmReg < TrapConfig::kLastReg) {

    for (int linkPair = 0; linkPair < mgkMaxLinkPairs; linkPair++) {
      if (FeeParam::extAliToAli(extali, linkPair, rocType, mcmList, mcmListSize) != 0) {
        int i = 0;
        while ((i < mcmListSize) && (mcmList[i] != -1)) {
          if (mcmList[i] == 127) {
            LOG(debug) << "broadcast write to " << mTrapConfig->getRegName((TrapConfig::TrapReg_t)mcmReg) << ": 0x" << hex << std::setw(8) << data;
            mTrapConfig->setTrapReg((TrapConfig::TrapReg_t)mcmReg, data, det);
          } else {
            LOG(debug) << "individual write to " << mTrapConfig->getRegName((TrapConfig::TrapReg_t)mcmReg) << " (" << (mcmList[i] >> 7) << "," << (mcmList[i] & 0x7F) << "): 0x" << hex << std::setw(8) << data;
            mTrapConfig->setTrapReg((TrapConfig::TrapReg_t)mcmReg, data, det, (mcmList[i] >> 7) & 0x7, (mcmList[i] & 0x7F));
          }
          i++;
        }
      }
    }
    return true;
  }
  // DMEM
  else if ((addr >= TrapConfig::mgkDmemStartAddress) &&
           (addr < (TrapConfig::mgkDmemStartAddress + TrapConfig::mgkDmemWords))) {
    for (int linkPair = 0; linkPair < mgkMaxLinkPairs; linkPair++) {
      if (FeeParam::extAliToAli(extali, linkPair, rocType, mcmList, mcmListSize) != 0) {
        int i = 0;
        while (i < mcmListSize && mcmList[i] != -1) {
          if (mcmList[i] == 127) {
            mTrapConfig->setDmem(addr, data, det, 0, 127);
          } else {
            mTrapConfig->setDmem(addr, data, det, mcmList[i] >> 7, mcmList[i] & 0x7f);
          }
          i++;
        }
      }
    }
    return true;
  } else if ((addr >= TrapConfig::mgkImemStartAddress) &&
             (addr < (TrapConfig::mgkImemStartAddress + TrapConfig::mgkImemWords))) {
    // IMEM is ignored for now
    return true;
  } else if ((addr >= TrapConfig::mgkDbankStartAddress) &&
             (addr < (TrapConfig::mgkDbankStartAddress + TrapConfig::mgkImemWords))) {
    // DBANK is ignored for now
    return true;
  } else {
    LOG(error) << "Writing to unhandled address 0x" << hex << std::setw(4) << addr;
    return false;
  }
}
