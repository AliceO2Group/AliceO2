#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
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
void dumpDetCTF(int ctfID, DetID det, TTree& treeIn, int ncolls)
{
  std::vector<o2::ctf::BufferType> buff;
  buff.resize(sizeof(C));
  C::readFromTree(buff, treeIn, det.getName(), ctfID);
  const auto ctf = C::getImage(buff.data());
  ctf.dump(det.getName(), ncolls);
}

void dumpCTF(const std::string& fnameIn, int ctfID = 0, const std::string selDet = "all", int ncolls = 20)
{
  std::unique_ptr<TFile> flIn(TFile::Open(fnameIn.c_str()));
  std::unique_ptr<TTree> treeIn((TTree*)flIn->Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
  if (treeIn->GetEntries() <= ctfID) {
    LOG(error) << "File " << fnameIn << " has only " << treeIn->GetEntries() << " entries, requested : " << ctfID;
    treeIn.reset();
    return;
  }

  o2::ctf::CTFHeader ctfHeader;
  if (!readFromTree(*treeIn, "CTFHeader", ctfHeader, ctfID)) {
    throw std::runtime_error("did not find CTFHeader");
  }

  DetID::mask_t detsTF = ctfHeader.detectors & DetID::getMask(selDet);
  if (detsTF.none()) {
    LOG(error) << "Nothing is selected with mask " << selDet << " CTF constains " << DetID::getNames(ctfHeader.detectors);
    treeIn.reset();
    return;
  }

  LOG(info) << ctfHeader;

  if (detsTF[DetID::ITS]) {
    dumpDetCTF<o2::itsmft::CTF>(ctfID, DetID::ITS, *treeIn, ncolls);
  }

  if (detsTF[DetID::MFT]) {
    dumpDetCTF<o2::itsmft::CTF>(ctfID, DetID::MFT, *treeIn, ncolls);
  }

  if (detsTF[DetID::TPC]) {
    dumpDetCTF<o2::tpc::CTF>(ctfID, DetID::TPC, *treeIn, ncolls);
  }

  if (detsTF[DetID::TRD]) {
    dumpDetCTF<o2::trd::CTF>(ctfID, DetID::TRD, *treeIn, ncolls);
  }

  if (detsTF[DetID::TOF]) {
    dumpDetCTF<o2::tof::CTF>(ctfID, DetID::TOF, *treeIn, ncolls);
  }

  if (detsTF[DetID::FT0]) {
    dumpDetCTF<o2::ft0::CTF>(ctfID, DetID::FT0, *treeIn, ncolls);
  }

  if (detsTF[DetID::FV0]) {
    dumpDetCTF<o2::fv0::CTF>(ctfID, DetID::FV0, *treeIn, ncolls);
  }

  if (detsTF[DetID::FDD]) {
    dumpDetCTF<o2::fdd::CTF>(ctfID, DetID::FDD, *treeIn, ncolls);
  }

  if (detsTF[DetID::MCH]) {
    dumpDetCTF<o2::mch::CTF>(ctfID, DetID::MCH, *treeIn, ncolls);
  }

  if (detsTF[DetID::MID]) {
    dumpDetCTF<o2::mid::CTF>(ctfID, DetID::MID, *treeIn, ncolls);
  }

  if (detsTF[DetID::ZDC]) {
    dumpDetCTF<o2::zdc::CTF>(ctfID, DetID::ZDC, *treeIn, ncolls);
  }

  if (detsTF[DetID::EMC]) {
    dumpDetCTF<o2::emcal::CTF>(ctfID, DetID::EMC, *treeIn, ncolls);
  }

  if (detsTF[DetID::PHS]) {
    dumpDetCTF<o2::phos::CTF>(ctfID, DetID::PHS, *treeIn, ncolls);
  }

  if (detsTF[DetID::CPV]) {
    dumpDetCTF<o2::cpv::CTF>(ctfID, DetID::CPV, *treeIn, ncolls);
  }

  if (detsTF[DetID::HMP]) {
    dumpDetCTF<o2::hmpid::CTF>(ctfID, DetID::HMP, *treeIn, ncolls);
  }

  if (detsTF[DetID::CTP]) {
    dumpDetCTF<o2::ctp::CTF>(ctfID, DetID::CTP, *treeIn, ncolls);
  }

  treeIn.reset();
}
