// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <CCDB/BasicCCDBManager.h>
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DetectorsCommonDataFormats/CTFDictHeader.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DataFormatsCTP/CTF.h"
#include "DataFormatsTRD/CTF.h"
#include "DataFormatsTOF/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsEMCAL/CTF.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsPHOS/CTF.h"
#include "DataFormatsZDC/CTF.h"
#include "DataFormatsMID/CTF.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsHMP/CTF.h"
#include "DataFormatsCPV/CTF.h"

#include <vector>
#include <TTree.h>

#endif

template <typename T>
size_t appendToTree(TTree& tree, const std::string brname, T& ptr)
{
  size_t s = 0;
  auto* br = tree.GetBranch(brname.c_str());
  auto* pptr = &ptr;
  if (br) {
    br->SetAddress(&pptr);
  } else {
    br = tree.Branch(brname.c_str(), &pptr);
  }
  int res = br->Fill();
  if (res < 0) {
    throw std::runtime_error(fmt::format("Failed to fill CTF branch {}", brname));
  }
  s += res;
  br->ResetAddress();
  return s;
}

template <typename C>
void conv(const std::vector<char>* buff)
{
  const auto& ctf = C::getImage(buff->data());
  const auto& ctfhead = ctf.getHeader();
  const auto& dictHead = (const o2::ctf::CTFDictHeader&)ctfhead;
  o2::ctf::CTFHeader header{};
  TFile fl(Form("ctf_dictionaryTree_%s_%u_0.root", dictHead.det.getName(), dictHead.dictTimeStamp), "recreate");
  TTree* tree = new TTree("ccdb_object", "O2 CTF dictionary");
  header.detectors.set(dictHead.det);
  ctf.appendToTree(*tree, dictHead.det.getName());
  appendToTree(*tree, "CTFHeader", header);
  tree->SetEntries(1);
  tree->Write(tree->GetName(), TObject::kSingleKey);
  delete tree;
  fl.Close();
  printf("Stored dictionary to %s\n", fl.GetName());
}

void convCTFDict(const char* det, long ts = 0, const char* ccdburl = "http://alice-ccdb.cern.ch")
{
  auto& cm = o2::ccdb::BasicCCDBManager::instance();
  if (ts > 0) {
    cm.setTimestamp(ts);
  }
  std::string pth = std::string(Form("%s/Calib/CTFDictionary", det));
  const auto* buff = cm.get<std::vector<char>>(pth);
  if (!buff) {
    printf("Failed to fetch from %s for %ld\n", pth.c_str(), cm.getTimestamp());
  }
  std::string dets{det};
  if (dets == "ITS" || dets == "MFT") {
    conv<o2::itsmft::CTF>(buff);
  } else if (dets == "TPC") {
    conv<o2::tpc::CTF>(buff);
  } else if (dets == "TRD") {
    conv<o2::trd::CTF>(buff);
  } else if (dets == "TOF") {
    conv<o2::tof::CTF>(buff);
  } else if (dets == "FT0") {
    conv<o2::ft0::CTF>(buff);
  } else if (dets == "FV0") {
    conv<o2::fv0::CTF>(buff);
  } else if (dets == "FDD") {
    conv<o2::fdd::CTF>(buff);
  } else if (dets == "EMC") {
    conv<o2::emcal::CTF>(buff);
  } else if (dets == "PHS") {
    conv<o2::phos::CTF>(buff);
  } else if (dets == "ZDC") {
    conv<o2::zdc::CTF>(buff);
  } else if (dets == "MID") {
    conv<o2::mid::CTF>(buff);
  } else if (dets == "MCH") {
    conv<o2::mch::CTF>(buff);
  } else if (dets == "HMP") {
    conv<o2::hmpid::CTF>(buff);
  } else if (dets == "CTP") {
    conv<o2::ctp::CTF>(buff);
  } else if (dets == "CPV") {
    conv<o2::cpv::CTF>(buff);
  }
}
