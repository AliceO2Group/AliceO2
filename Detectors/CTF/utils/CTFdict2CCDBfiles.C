#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsCommonDataFormats/CTFDictHeader.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTRD/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsTOF/CTF.h"
#include "DataFormatsMID/CTF.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsEMCAL/CTF.h"
#include "DataFormatsPHOS/CTF.h"
#include "DataFormatsCPV/CTF.h"
#include "DataFormatsZDC/CTF.h"
#include "DataFormatsHMP/CTF.h"
#include "DataFormatsCTP/CTF.h"
#include <fmt/format.h>
#endif

using DetID = o2::detectors::DetID;

template <typename T>
bool readFromTree(TTree& tree, const std::string brname, T& dest, int ev = 0)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (br && br->GetEntries() > ev) {
    auto* ptr = &dest;
    br->SetAddress(&ptr);
    br->GetEntry(ev);
    br->ResetAddress();
    return true;
  }
  return false;
}

template <typename C>
void extractDictionary(TTree& tree, o2::detectors::DetID det, DetID::mask_t detMask)
{
  std::vector<char> bufVec;
  if (!detMask[det]) {
    return;
  }
  o2::ctf::CTFHeader ctfHeader;
  readFromTree(tree, "CTFHeader", ctfHeader);
  if (!ctfHeader.detectors[det]) {
    LOGP(warning, "Dictionary for {} was requested but absent", det.getName());
    return;
  }
  C::readFromTree(bufVec, tree, det.getName());
  auto& dictHeader = static_cast<o2::ctf::CTFDictHeader&>(C::get(bufVec.data())->getHeader());
  dictHeader.det = det; // impose detector, since in early versions it is absent
  std::string outName = fmt::format("ctfdict_{}_v{}.{}_{}.root", det.getName(), int(dictHeader.majorVersion), int(dictHeader.minorVersion), dictHeader.dictTimeStamp);
  TFile flout(outName.c_str(), "recreate");
  flout.WriteObject(&bufVec, o2::base::NameConf::CCDBOBJECT.data());
  flout.WriteObject(&dictHeader, fmt::format("ctf_dict_header_{}", det.getName()).c_str());
  flout.Close();
  LOG(info) << "Wrote " << dictHeader.asString() << " to " << outName;
}

// This macro allows to convert tree-based CTF dictionary (produced by the ctf-writer-workflow) to per-detector files with plain vector, suitable for the CCDB use.
void CTFdict2CCDBfiles(const std::string& fname = "ctf_dictionary.root", const std::string dets = "all")
{
  std::string allowedDetectors = "ITS,TPC,TRD,TOF,PHS,CPV,EMC,HMP,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP";
  auto detMask = DetID::getMask(dets) & DetID::getMask(allowedDetectors);

  std::unique_ptr<TFile> dictFile(TFile::Open(fname.c_str()));
  if (!dictFile) {
    LOG(error) << "Failed to open CTF dictionary file " << fname;
    return;
  }
  std::unique_ptr<TTree> tree((TTree*)dictFile->Get(std::string(o2::base::NameConf::CTFDICT).c_str()));
  if (!tree) {
    LOG(error) << "Did not find CTF dictionary tree in " << fname;
    return;
  }
  extractDictionary<o2::itsmft::CTF>(*tree, DetID::ITS, detMask);
  extractDictionary<o2::itsmft::CTF>(*tree, DetID::MFT, detMask);
  extractDictionary<o2::emcal::CTF>(*tree, DetID::EMC, detMask);
  extractDictionary<o2::hmpid::CTF>(*tree, DetID::HMP, detMask);
  extractDictionary<o2::phos::CTF>(*tree, DetID::PHS, detMask);
  extractDictionary<o2::tpc::CTF>(*tree, DetID::TPC, detMask);
  extractDictionary<o2::trd::CTF>(*tree, DetID::TRD, detMask);
  extractDictionary<o2::ft0::CTF>(*tree, DetID::FT0, detMask);
  extractDictionary<o2::fv0::CTF>(*tree, DetID::FV0, detMask);
  extractDictionary<o2::fdd::CTF>(*tree, DetID::FDD, detMask);
  extractDictionary<o2::tof::CTF>(*tree, DetID::TOF, detMask);
  extractDictionary<o2::mid::CTF>(*tree, DetID::MID, detMask);
  extractDictionary<o2::mch::CTF>(*tree, DetID::MCH, detMask);
  extractDictionary<o2::cpv::CTF>(*tree, DetID::CPV, detMask);
  extractDictionary<o2::zdc::CTF>(*tree, DetID::ZDC, detMask);
  extractDictionary<o2::ctp::CTF>(*tree, DetID::CTP, detMask);

  tree.reset();
  dictFile.reset();
}
