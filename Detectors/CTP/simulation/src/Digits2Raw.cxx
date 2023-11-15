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

/// \file Digits2Raw.cxx
/// \author Roman Lietava

#include "CTPSimulation/Digits2Raw.h"
#include <fairlogger/Logger.h>
#include "CommonUtils/StringUtils.h"

using namespace o2::ctp;

/// CTP digits (see Digits.h) are inputs from trigger detectors.
/// GBT digit is temporary item: 80 bits long = 12 bits of BCid and payload (see digit2GBTdigit)
/// CTP digit -> GBTdigit CTPInt Record, GBTdigit TrigClass Record
/// GBT didit X -> CRU  raw data
/// X= CTPInt Record to link 0
/// X= TrigClass Record to link 1

void Digits2Raw::init()
{
  //
  // Register links
  //
  std::string outd = mOutDir;
  if (outd.back() != '/') {
    outd += '/';
  }
  LOG(info) << "Raw outpud dir:" << mOutDir;
  //
  LOG(info) << "Raw Data padding:" << mPadding;
  // Interaction Record
  int ilink = 0;
  uint64_t feeID = getFEEIDIR();
  std::string outFileLink0 = mOutputPerLink ? fmt::format("{}{}_feeid{}.raw", outd, mCTPRawDataFileName, feeID) : fmt::format("{}{}.raw", outd, mCTPRawDataFileName);
  mWriter.registerLink(feeID, mCruID, ilink, mEndPointID, outFileLink0);
  // Trigger Class record
  ilink = 1;
  feeID = getFEEIDTC();
  std::string outFileLink1 = mOutputPerLink ? fmt::format("{}{}_feeid{}.raw", outd, mCTPRawDataFileName, feeID) : fmt::format("{}{}.raw", outd, mCTPRawDataFileName);
  mWriter.registerLink(feeID, mCruID, ilink, mEndPointID, outFileLink1);
  // ilink = 2: HBMap, Counters - tbd
  mWriter.setEmptyPageCallBack(this);
}
void Digits2Raw::processDigits(const std::string& fileDigitsName)
{
  std::unique_ptr<TFile> digiFile(TFile::Open(fileDigitsName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(fatal) << "Failed to open input digits file " << fileDigitsName;
    return;
  }
  LOG(info) << "Processing digits to raw file:" << fileDigitsName;
  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(fatal) << "Failed to get digits tree";
    return;
  }
  std::vector<o2::ctp::CTPDigit> CTPDigits, *fCTPDigitsPtr = &CTPDigits;
  if (digiTree->GetBranch("CTPDigits")) {
    digiTree->SetBranchAddress("CTPDigits", &fCTPDigitsPtr);
  } else {
    LOG(fatal) << "Branch CTPDigits is missing";
    return;
  }
  o2::InteractionRecord intRec = {0, 0};
  // Get first orbit
  uint32_t orbit0 = 0;
  bool firstorbit = 1;
  // Add all CTPdigits for given orbit
  LOG(info) << "Number of entries: " << digiTree->GetEntries();
  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    int nbc = CTPDigits.size();
    LOG(debug) << "Entry " << ient << " : " << nbc << " BCs stored";
    std::vector<gbtword80_t> hbfIR;
    std::vector<gbtword80_t> hbfTC;
    for (auto const& ctpdig : CTPDigits) {
      LOG(debug) << ctpdig.intRecord.bc << " bc all orbit " << ctpdig.intRecord.orbit;
      if ((orbit0 == ctpdig.intRecord.orbit) || firstorbit) {
        if (firstorbit == true) {
          firstorbit = false;
          orbit0 = ctpdig.intRecord.orbit;
          LOG(info) << "First orbit:" << orbit0;
        }
        LOG(debug) << ctpdig.intRecord.orbit << " orbit bc " << ctpdig.intRecord.bc;
        gbtword80_t gbtdigIR;
        gbtword80_t gbtdigTC;
        digit2GBTdigit(gbtdigIR, gbtdigTC, ctpdig);
        LOG(debug) << "ir:" << gbtdigIR << " " << (gbtdigIR.to_ullong() & 0xfff);
        LOG(debug) << "tr:" << gbtdigTC;
        hbfIR.push_back(gbtdigIR);
        hbfTC.push_back(gbtdigTC);
      } else {
        std::vector<char> buffer;
        LOG(info) << "Packing orbit:" << orbit0 << " hbfIR:" << hbfIR.size() << " hbfTC:" << hbfTC.size();
        intRec.orbit = orbit0;
        if (mZeroSuppressedIntRec == true) {
          buffer = digits2HBTPayload(hbfIR, NIntRecPayload);
        } else {
          std::vector<gbtword80_t> hbfIRnonZS = addEmptyBC(hbfIR);
          buffer = digits2HBTPayload(hbfIRnonZS, NIntRecPayload);
        }
        // add data for IR
        LOG(debug) << "IR buffer size:" << buffer.size() << ":";
        mWriter.addData(getFEEIDIR(), mCruID, GBTLinkIDIntRec, mEndPointID, intRec, buffer);
        // add data for Trigger Class Record
        buffer.clear();
        buffer = digits2HBTPayload(hbfTC, NClassPayload);
        LOG(debug) << "TC buffer size:" << buffer.size() << ":";
        mWriter.addData(getFEEIDTC(), mCruID, GBTLinkIDClassRec, mEndPointID, intRec, buffer);
        //
        orbit0 = ctpdig.intRecord.orbit;
        hbfIR.clear();
        hbfTC.clear();
        LOG(debug) << ctpdig.intRecord.orbit << " orbit bc " << ctpdig.intRecord.bc;
        gbtword80_t gbtdigIR;
        gbtword80_t gbtdigTC;
        digit2GBTdigit(gbtdigIR, gbtdigTC, ctpdig);
        LOG(debug) << "ir:" << gbtdigIR;
        LOG(debug) << "tr:" << gbtdigTC;
        hbfIR.push_back(gbtdigIR);
        hbfTC.push_back(gbtdigTC);
      }
      intRec = ctpdig.intRecord;
    }
    // Last orbit in record
    std::vector<char> buffer;
    LOG(info) << "Packing orbit last:" << orbit0;
    intRec.orbit = orbit0;
    if (mZeroSuppressedIntRec == true) {
      buffer = digits2HBTPayload(hbfIR, NIntRecPayload);
    } else {
      std::vector<gbtword80_t> hbfIRnonZS = addEmptyBC(hbfIR);
      buffer = digits2HBTPayload(hbfIRnonZS, NIntRecPayload);
    }
    // add data for IR
    LOG(debug) << "IR buffer size:" << buffer.size() << " orbit:" << intRec.orbit;
    mWriter.addData(getFEEIDIR(), mCruID, GBTLinkIDIntRec, mEndPointID, intRec, buffer);
    // add data for Trigger Class Record
    buffer.clear();
    buffer = digits2HBTPayload(hbfTC, NClassPayload);
    LOG(debug) << "TC buffer size:" << buffer.size() << " orbit:" << intRec.orbit;
    mWriter.addData(getFEEIDTC(), mCruID, GBTLinkIDClassRec, mEndPointID, intRec, buffer);
    //
    //orbit0 = ctpdig.intRecord.orbit;
    firstorbit = true;
    hbfIR.clear();
    hbfTC.clear();
  }
}
void Digits2Raw::emptyHBFMethod(const header::RDHAny* rdh, std::vector<char>& toAdd) const
{
  // TriClassRecord data zero suppressed
  // CTP INteraction Data
  if (((o2::raw::RDHUtils::getFEEID(rdh) & 0xf00) >> 8) == GBTLinkIDIntRec) {
    if (mZeroSuppressedIntRec == false) {
      toAdd.clear();
      std::vector<gbtword80_t> digits;
      for (uint32_t i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
        gbtword80_t dig = i;
        digits.push_back(dig);
      }
      toAdd = digits2HBTPayload(digits, NIntRecPayload);
    }
  }
}
std::vector<char> Digits2Raw::digits2HBTPayload(const gsl::span<gbtword80_t> digits, uint32_t Npld) const
{
  std::vector<char> toAdd;
  int countBytes = 0;
  uint32_t size_gbt = 0;
  gbtword80_t gbtword;
  gbtword80_t gbtsend;
  bool valid;
  for (auto const& dig : digits) {
    valid = makeGBTWord(dig, gbtword, size_gbt, Npld, gbtsend);
    LOG(debug) << Npld << " digit:" << dig << " " << (dig.to_ulong() & 0xfff) << " ";
    LOG(debug) << "gbt  :" << gbtsend << " valid:" << valid;
    if (valid == true) {
      for (uint32_t i = 0; i < NGBT; i += 8) {
        uint32_t w = 0;
        for (uint32_t j = 0; j < 8; j++) {
          w += (1 << j) * gbtsend[i + j];
        }
        countBytes++;
        char c = w;
        toAdd.push_back(c);
      }
      if (mPadding) {
        // Pad zeros up to 128 bits
        uint32_t NZeros = (o2::raw::RDHUtils::GBTWord128 * 8 - NGBT) / 8;
        for (uint32_t i = 0; i < NZeros; i++) {
          char c = 0;
          toAdd.push_back(c);
        }
      }
    }
  }
  // add what is left: maybe never left anything - tbc
  LOG(debug) << size_gbt << " size valid " << valid;
  LOG(debug) << "gbtword:" << gbtword;
  LOG(debug) << "gbtsend:" << gbtsend;
  if (size_gbt > 0) {
    // LOG(info) << "Adding left over.";
    gbtword80_t gbtsend = gbtword;
    for (uint32_t i = 0; i < NGBT; i += 8) {
      uint32_t w = 0;
      for (uint32_t j = 0; j < 8; j++) {
        w += (1 << j) * gbtsend[i + j];
      }
      countBytes++;
      char c = w;
      toAdd.push_back(c);
    }
    // Pad zeros up to 128 bits
    if (mPadding) {
      uint32_t NZeros = (o2::raw::RDHUtils::GBTWord128 * 8 - NGBT) / 8;
      for (uint32_t i = 0; i < NZeros; i++) {
        char c = 0;
        toAdd.push_back(c);
      }
    }
  }
  return std::move(toAdd);
}
// Adding payload of size < NGBT to GBT words of size NGBT
// gbtsend valid only when return 1
bool Digits2Raw::makeGBTWord(const gbtword80_t& pld, gbtword80_t& gbtword, uint32_t& size_gbt, uint32_t Npld, gbtword80_t& gbtsend) const
{
  bool valid = false;
  //printBitset(gbtword,"GBTword");
  gbtword |= (pld << size_gbt);
  if ((size_gbt + Npld) < NGBT) {
    size_gbt += Npld;
  } else {
    // sendData
    //printBitset(gbtword,"Sending");
    gbtsend = gbtword;
    gbtword = pld >> (NGBT - size_gbt);
    size_gbt = size_gbt + Npld - NGBT;
    valid = true;
  }
  return valid;
}
int Digits2Raw::digit2GBTdigit(gbtword80_t& gbtdigitIR, gbtword80_t& gbtdigitTR, const CTPDigit& digit)
{
  //
  // CTP Interaction record (CTP inputs)
  //
  gbtdigitIR = 0;
  gbtdigitIR = (digit.CTPInputMask).to_ullong() << 12;
  gbtdigitIR |= digit.intRecord.bc;
  //
  // Trig Classes
  //
  gbtdigitTR = 0;
  //gbtdigitTR = (digit.CTPClassMask).to_ullong() << 12;
  gbtdigitTR |= digit.intRecord.bc;
  for (int i = 0; i < CTP_NCLASSES; i++) {
    gbtdigitTR[i + 12] = digit.CTPClassMask[i];
  }
  return 0;
}
std::vector<gbtword80_t> Digits2Raw::addEmptyBC(std::vector<gbtword80_t>& hbfIRZS)
{
  std::vector<gbtword80_t> hbfIRnonZS;
  if (hbfIRZS.size() == 0) {
    LOG(error) << "Int record with zero size not expected here.";
    return hbfIRnonZS;
  }
  uint32_t bcnonzero = 0;
  if (hbfIRZS[0] != 0) {
    gbtword80_t bs = 0;
    hbfIRnonZS.push_back(bs);
  }
  for (auto const& item : hbfIRZS) {
    uint32_t bcnonzeroNext = (item.to_ulong()) & 0xfff;
    for (int i = (bcnonzero + 1); i < bcnonzeroNext; i++) {
      gbtword80_t bs = i;
      hbfIRnonZS.push_back(bs);
    }
    bcnonzero = bcnonzeroNext;
    hbfIRnonZS.push_back(item);
  }
  for (int i = (bcnonzero + 1); i < 3564; i++) {
    gbtword80_t bs = i;
    hbfIRnonZS.push_back(bs);
  }
  return hbfIRnonZS;
}
void Digits2Raw::printDigit(std::string text, const gbtword80_t& dig) const
{
  int bcid = 0;
  uint64_t payload1 = 0;
  uint64_t payload2 = 0;
  std::cout << text;
  for (int i = 0; i < 12; i++) {
    bcid += dig[i] << i;
  }
  for (uint64_t i = 0; i < 64; i++) {
    payload1 += uint64_t(dig[i]) << i;
  }
  for (uint64_t i = 64; i < NGBT; i++) {
    payload2 += uint64_t(dig[i]) << (i - 64ull);
  }
  std::cout << "BCID:" << std::hex << bcid << " " << payload2 << payload1 << std::endl;
}
void Digits2Raw::dumpRawData(std::string filename)
{
  std::ifstream input(filename, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
  for (auto const& cc : buffer) {
  }
}
