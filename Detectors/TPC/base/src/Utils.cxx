// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cmath>
#include <regex>
#include <string>
#include <fmt/format.h>
#include <fmt/printf.h>

#include "TSystem.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFile.h"
#include "TChain.h"

#include "Framework/Logger.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Utils.h"

using namespace o2::tpc;

/// Inspired by https://stackoverflow.com/questions/9435385/split-a-string-using-c11
/// could be optimized for speed, see e.g. https://stackoverflow.com/questions/14205096/c11-regex-slower-than-python
const std::vector<std::string> utils::tokenize(const std::string_view input, const std::string_view pattern)
{
  // passing -1 as the submatch index parameter performs splitting
  std::regex re(pattern.data());
  std::cregex_token_iterator
    first{input.begin(), input.end(), re, -1},
    last;
  return {first, last};
}

TH1* utils::getBinInfoXY(int& binx, int& biny, float& bincx, float& bincy)
{
  TObject* select = gPad->GetSelected();
  if (!select) {
    return nullptr;
  }
  if (!select->InheritsFrom("TH2")) {
    gPad->SetUniqueID(0);
    return nullptr;
  }

  TH1* h = (TH1*)select;
  gPad->GetCanvas()->FeedbackMode(kTRUE);

  const int px = gPad->GetEventX();
  const int py = gPad->GetEventY();
  const float xx = gPad->AbsPixeltoX(px);
  const float x = gPad->PadtoX(xx);
  const float yy = gPad->AbsPixeltoY(py);
  const float y = gPad->PadtoX(yy);
  binx = h->GetXaxis()->FindBin(x);
  biny = h->GetYaxis()->FindBin(y);
  bincx = h->GetXaxis()->GetBinCenter(binx);
  bincy = h->GetYaxis()->GetBinCenter(biny);
  //printf("binx, biny: %d %d\n",binx,biny);

  return h;
}

void utils::addFECInfo()
{
  using namespace o2::tpc;
  const int event = gPad->GetEvent();
  if (event != 51) {
    return;
  }

  int binx, biny;
  float bincx, bincy;
  TH1* h = utils::getBinInfoXY(binx, biny, bincx, bincy);
  if (!h) {
    return;
  }

  const float binValue = h->GetBinContent(binx, biny);
  const int row = int(std::floor(bincx));
  const int cpad = int(std::floor(bincy));

  const auto& mapper = Mapper::instance();

  const int roc = h->GetUniqueID();
  if (roc < 0 || roc >= (int)ROC::MaxROC)
    return;
  if (row < 0 || row >= (int)mapper.getNumberOfRowsROC(roc))
    return;
  const int nPads = mapper.getNumberOfPadsInRowROC(roc, row);
  const int pad = cpad + nPads / 2;
  //printf("row %d, cpad %d, pad %d, nPads %d\n", row, cpad, pad, nPads);
  if (pad < 0 || pad >= (int)nPads) {
    return;
  }
  const int channel = mapper.getPadNumberInROC(PadROCPos(roc, row, pad));

  const auto& fecInfo = mapper.getFECInfo(PadROCPos(roc, row, pad));

  const std::string title = fmt::format(
    "#splitline{{#lower[.1]{{#scale[.5]{{"
    "{}{:02d} ({:02d}) row: {:02d}, pad: {:03d}, globalpad: {:05d} (in roc)"
    "}}}}}}{{#scale[.5]{{FEC: "
    "{:02d}, Chip: {:02d}, Chn: {:02d}, Value: {:.3f}"
    "}}}}",
    (roc / 18 % 2 == 0) ? "A" : "C", roc % 18, roc, row, pad, channel, fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel(), binValue);

  h->SetTitle(title.data());
}

void utils::saveCanvases(TObjArray& arr, std::string_view outDir, std::string_view types, std::string_view rootFileName)
{
  for (auto c : arr) {
    utils::saveCanvas(*static_cast<TCanvas*>(c), outDir, types);
  }

  if (rootFileName.size()) {
    std::unique_ptr<TFile> outFile(TFile::Open(fmt::format("{}/NoiseAndPedestalCanvases.root", outDir).data(), "recreate"));
    arr.Write(arr.GetName(), TObject::kSingleKey);
    outFile->Close();
  }
}

void utils::saveCanvases(std::vector<TCanvas*> canvases, std::string_view outDir, std::string_view types, std::string_view rootFileName)
{
  TObjArray arr;
  for (auto c : canvases) {
    arr.Add(c);
  }

  saveCanvases(arr, outDir, types, rootFileName);
}

void utils::saveCanvas(TCanvas& c, std::string_view outDir, std::string_view types)
{
  const auto typesVec = tokenize(types, ",");
  for (const auto& type : typesVec) {
    c.SaveAs(fmt::format("{}/{}.{}", outDir, c.GetName(), type).data());
  }
}

std::vector<CalPad*> utils::readCalPads(const std::string_view fileName, const std::vector<std::string>& calPadNames)
{
  std::vector<CalPad*> calPads(calPadNames.size());

  std::unique_ptr<TFile> file(TFile::Open(fileName.data()));
  if (!file || !file->IsOpen() || file->IsZombie()) {
    return calPads;
  }

  for (size_t iCalPad = 0; iCalPad < calPadNames.size(); ++iCalPad) {
    file->GetObject(calPadNames[iCalPad].data(), calPads[iCalPad]);
  }

  return calPads;
}

std::vector<CalPad*> utils::readCalPads(const std::string_view fileName, const std::string_view calPadNames)
{
  auto calPadNamesVec = tokenize(calPadNames, ",");
  return readCalPads(fileName, calPadNamesVec);
}

//______________________________________________________________________________
void utils::mergeCalPads(std::string_view outputFileName, std::string_view inputFileNames, std::string_view calPadNames)
{
  using namespace o2::tpc;

  auto calPadNamesVec = utils::tokenize(calPadNames, ",");

  std::string_view cmd = "ls";
  if (inputFileNames.rfind(".root") == std::string_view::npos) {
    cmd = "cat";
  }
  auto files = gSystem->GetFromPipe(TString::Format("%s %s", cmd.data(), inputFileNames.data()));
  std::unique_ptr<TObjArray> arrFiles(files.Tokenize("\n"));

  std::vector<CalPad*> mergedCalPads;

  for (auto ofile : *arrFiles) {
    auto calPads = utils::readCalPads(ofile->GetName(), calPadNamesVec);
    if (!calPads.size()) {
      continue;
    }
    if (!mergedCalPads.size()) {
      mergedCalPads = calPads;
    } else {
      for (size_t iCalPad = 0; iCalPad < calPads.size(); ++iCalPad) {
        auto calPadName = calPadNamesVec[iCalPad];
        auto calPadMerged = mergedCalPads[iCalPad];
        calPadMerged->setName(calPadName);
        auto calPadToMerge = calPads[iCalPad];

        *calPadMerged += *calPadToMerge;

        delete calPadToMerge;
      }
    }
  }

  std::unique_ptr<TFile> outFile(TFile::Open(outputFileName.data(), "recreate"));
  for (auto calPad : mergedCalPads) {
    outFile->WriteObject(calPad, calPad->getName().data());
  }
}

//______________________________________________________________________________
TChain* utils::buildChain(std::string_view command, std::string_view treeName, std::string_view treeTitle)
{
  const TString files = gSystem->GetFromPipe(command.data());
  std::unique_ptr<TObjArray> arrFiles(files.Tokenize("\n"));
  if (!arrFiles->GetEntriesFast()) {
    LOGP(error, "command '{}' did not return results", command);
    return nullptr;
  }

  auto c = new TChain(treeName.data(), treeTitle.data());
  for (const auto o : *arrFiles) {
    LOGP(info, "Adding file '{}'", o->GetName());
    c->AddFile(o->GetName());
  }

  return c;
}

