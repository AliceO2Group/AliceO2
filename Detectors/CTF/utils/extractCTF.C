#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DetectorsCommonDataFormats/NameConf.h"
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
void writeToTree(TTree& tree, const std::vector<o2::ctf::BufferType>& buff, DetID det)
{
  const auto ctfImage = C::getImage(buff.data());
  ctfImage.appendToTree(tree, det.getName());
}

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
void extractDetCTF(int ctfID, DetID det, TTree& treeIn, TTree& treeOut)
{
  std::vector<o2::ctf::BufferType> buff;
  buff.resize(sizeof(C));
  C::readFromTree(buff, treeIn, det.getName(), ctfID);
  writeToTree<C>(treeOut, buff, det);
}

void extractCTF(int ctfID,
                const std::string& fnameIn,
                const std::string& fnameOut,
                const std::string selDet = "all")
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

  LOG(info) << ctfHeader;
  DetID::mask_t detsTF = ctfHeader.detectors & DetID::getMask(selDet);
  if (detsTF.none()) {
    LOG(error) << "Nothing is selected with mask " << selDet << " CTF constains " << DetID::getNames(ctfHeader.detectors);
    treeIn.reset();
    return;
  }
  ctfHeader.detectors = detsTF;

  TFile flOut(fnameOut.c_str(), "recreate");
  std::unique_ptr<TTree> treeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");

  if (detsTF[DetID::ITS]) {
    extractDetCTF<o2::itsmft::CTF>(ctfID, DetID::ITS, *treeIn, *treeOut);
  }

  if (detsTF[DetID::MFT]) {
    extractDetCTF<o2::itsmft::CTF>(ctfID, DetID::MFT, *treeIn, *treeOut);
  }

  if (detsTF[DetID::TPC]) {
    extractDetCTF<o2::tpc::CTF>(ctfID, DetID::TPC, *treeIn, *treeOut);
  }

  if (detsTF[DetID::TRD]) {
    extractDetCTF<o2::trd::CTF>(ctfID, DetID::TRD, *treeIn, *treeOut);
  }

  if (detsTF[DetID::TOF]) {
    extractDetCTF<o2::tof::CTF>(ctfID, DetID::TOF, *treeIn, *treeOut);
  }

  if (detsTF[DetID::FT0]) {
    extractDetCTF<o2::ft0::CTF>(ctfID, DetID::FT0, *treeIn, *treeOut);
  }

  if (detsTF[DetID::FV0]) {
    extractDetCTF<o2::fv0::CTF>(ctfID, DetID::FV0, *treeIn, *treeOut);
  }

  if (detsTF[DetID::FDD]) {
    extractDetCTF<o2::fdd::CTF>(ctfID, DetID::FDD, *treeIn, *treeOut);
  }

  if (detsTF[DetID::MCH]) {
    extractDetCTF<o2::mch::CTF>(ctfID, DetID::MCH, *treeIn, *treeOut);
  }

  if (detsTF[DetID::MID]) {
    extractDetCTF<o2::mid::CTF>(ctfID, DetID::MID, *treeIn, *treeOut);
  }

  if (detsTF[DetID::ZDC]) {
    extractDetCTF<o2::zdc::CTF>(ctfID, DetID::ZDC, *treeIn, *treeOut);
  }

  if (detsTF[DetID::EMC]) {
    extractDetCTF<o2::emcal::CTF>(ctfID, DetID::EMC, *treeIn, *treeOut);
  }

  if (detsTF[DetID::PHS]) {
    extractDetCTF<o2::phos::CTF>(ctfID, DetID::PHS, *treeIn, *treeOut);
  }

  if (detsTF[DetID::CPV]) {
    extractDetCTF<o2::cpv::CTF>(ctfID, DetID::CPV, *treeIn, *treeOut);
  }

  if (detsTF[DetID::HMP]) {
    extractDetCTF<o2::hmpid::CTF>(ctfID, DetID::HMP, *treeIn, *treeOut);
  }

  if (detsTF[DetID::CTP]) {
    extractDetCTF<o2::ctp::CTF>(ctfID, DetID::CTP, *treeIn, *treeOut);
  }

  appendToTree(*treeOut, "CTFHeader", ctfHeader);

  treeOut->SetEntries(1);

  LOG(info) << "Wrote CTFs of entry " << ctfID << " for " << DetID::getNames(ctfHeader.detectors) << " to " << fnameOut;

  treeOut->Write();
  treeOut.reset();
  flOut.Close();

  treeIn.reset();
}
