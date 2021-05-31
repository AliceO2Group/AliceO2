// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits2Raw.cxx
/// \author Roman Lietava


#include "CTPSimulation/Digits2Raw.h"
#include "FairLogger.h"
#include "CommonUtils/StringUtils.h"


using namespace o2::ctp;

//ClassImp(Digits2Raw);

void Digits2Raw::init()
{
  // CTP Configuration
    if(mCCDBServer.empty()) {
    LOG(FATAL) << "CTP digitizer: CCDB server is not set";
  } else {
    LOG(INFO) << "CTP digitizer:: CCDB server:" << mCCDBServer;  
  }
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBServer);
  mCTPConfiguration = mgr.get<CTPConfiguration>(o2::ctp::CCDBPathCTPConfig);
  //
  // Register links
  //
  std::string outd = mOutDir;
  if (outd.back() != '/') {
    outd += '/';
  }  
  mLinkID = uint32_t(0);
  mCruID = uint16_t(0);
  mEndPointID = uint32_t(0);
  for (int ilink = 0; ilink < mNLinks; ilink++) {
    mFeeID = uint64_t(ilink);
    std::string outFileLink = mOutputPerLink ? o2::utils::Str::concat_string(outd, "ctp_link", std::to_string(ilink), ".raw") : o2::utils::Str::concat_string(outd, "ctp.raw");
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }
  mWriter.setEmptyPageCallBack(this);
}
void Digits2Raw::processDigits(const std::string& fileDigitsName)
{
  std::unique_ptr<TFile> digiFile(TFile::Open(fileDigitsName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(FATAL) << "Failed to open input digits file " << fileDigitsName;
    return;
  }
  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(FATAL) << "Failed to get digits tree";
    return;
  }
  std::vector<o2::ctp::CTPDigit> CTPDigits, *fCTPDigitsPtr = &CTPDigits;
  if (digiTree->GetBranch("CTPDigits")) {
    digiTree->SetBranchAddress("CTPDigits", &fCTPDigitsPtr);
  } else {
    LOG(FATAL) << "Branch CTPDigits is missing";
    return;
  }
  // Get first orbit
  digiTree->GetEntry(0);
  uint32_t orbit0 ;
  if(CTPDigits.size()>0) {
    orbit0 = CTPDigits[0].intRecord.orbit;
  } else {
    LOG(FATAL) << "Branch CTPDigits there but no entries";
    return;
  }
  // Add all CTPdigits for given orbit
  for (int ient = 1; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    int nbc = CTPDigits.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    std::vector<std::bitset<NGBT>> hbf;
    for(auto const& ctpdig: CTPDigits) {
      if(orbit0 == ctpdig.intRecord.orbit) {
        std::bitset<NGBT> gbtdig;
        digit2GBTdigit(gbtdig,ctpdig);
        hbf.push_back(gbtdig);
      } else {
        std::vector<char> buffer;
        buffer = digits2HBTPayload(hbf,NIntRecPayload);
        std::string PLTrailer;
        std::memcpy(buffer.data() + buffer.size() - o2::raw::RDHUtils::GBTWord, PLTrailer.c_str(), o2::raw::RDHUtils::GBTWord);
        orbit0 = ctpdig.intRecord.orbit;
      }
    }
  }
}
void Digits2Raw::emptyHBFMethod(const header::RDHAny* rdh, std::vector<char>& toAdd) const
{
  // TriClassRecord data zero suppressed  
  // CTP INteraction Data
  if(mActiveLink == CRULinkIDIntRec) {
    if(mZeroSuppressedIntRec == false)
    {
      toAdd.clear();
      std::vector<std::bitset<NGBT>> digits;
      for(uint32_t i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
          std::bitset<NGBT> dig = i;
          digits.push_back(dig);
      }
      toAdd = digits2HBTPayload(digits,NIntRecPayload);
    }
  }
  toAdd.resize(o2::raw::RDHUtils::GBTWord);
  std::memcpy(toAdd.data(), HBFEmpty.c_str(), o2::raw::RDHUtils::GBTWord);
}
std::vector<char> Digits2Raw::digits2HBTPayload(const gsl::span<std::bitset<NGBT>> digits, uint32_t Npld) const
{
  std::vector<char> toAdd;
  uint32_t size_gbt=0;
  std::bitset<NGBT> gbtword;
  for(auto const &dig: digits)
  {
      if(makeGBTWord(dig, gbtword, size_gbt, Npld) == true)
      {
          for(uint32_t i=0; i< NGBT; i+=8)
          {
              uint32_t w =0;
              for(uint32_t j = 0 ;j < 8 ; j++)
              {
                  w += (1<<j) * gbtword[i+j];
              }
              char c = w;    
              toAdd.push_back(c);
          }
          // Pad zeros up to 128 bits
          uint32_t NZeros = (o2::raw::RDHUtils::GBTWord*8 - NGBT)/8;
          for(uint32_t i=0; i< NZeros; i++)
          {
            char c=0;
            toAdd.push_back(c);
          }
      }
  }
  return std::move(toAdd);
} 
bool Digits2Raw::makeGBTWord(const std::bitset<NGBT> &pld,std::bitset<NGBT> &gbtword, uint32_t& size_gbt,uint32_t Npld) const
{
  bool valid = false;
  //printBitset(gbtword,"GBTword");
  gbtword |= (pld << size_gbt);
  if((size_gbt+Npld) < NGBT)
  {
      size_gbt += Npld;
  }
  else
  {
      // sendData
      //printBitset(gbtword,"Sending");
      gbtword = pld >> (NGBT - size_gbt);
      size_gbt = size_gbt + Npld - NGBT;
      valid = true;
  }
  return valid;
}
int Digits2Raw::digit2GBTdigit(std::bitset<NGBT>& gbtdigit, const CTPDigit& digit)
{
  uint64_t gbtmask=0;
  uint64_t digmask=(digit.CTPInputMask).to_ullong();
  // Also CTP Detector Input configuration shoiuld be employed
  uint64_t allT0s =  (CTP_INPUTMASK_FT0.second).to_ullong() >> CTP_INPUTMASK_FT0.first;
  if(digmask & allT0s) {
    // assuming T0A - 1st bit
    if(digmask & (1ull >> CTP_INPUTMASK_FT0.first)) {
      gbtmask |= mCTPConfiguration->getInputMask("T0A");
    }
    // assuming T0B - 2nd bit
    if(digmask & (2ull >> CTP_INPUTMASK_FT0.first)) {
      gbtmask |= mCTPConfiguration->getInputMask("T0B");
    }
  }
  uint64_t allV0s =  (CTP_INPUTMASK_FV0.second).to_ullong() >> CTP_INPUTMASK_FV0.first;
  if(digmask & allV0s) {
      // assuming V0A - 1st bit
    if(digmask & (1ull >> CTP_INPUTMASK_FV0.first)) {
      gbtmask |= mCTPConfiguration->getInputMask("V0A");
    }
    // assuming V0B - 2nd bit
    if(digmask & (2ull >> CTP_INPUTMASK_FV0.first)) {
      gbtmask |= mCTPConfiguration->getInputMask("V0B");
    }
  }
  gbtdigit = gbtmask>>12;
  gbtdigit |= digit.intRecord.bc;
  return 0;
}
