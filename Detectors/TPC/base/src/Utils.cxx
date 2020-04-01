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

#include "TObject.h"
#include "TObjArray.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFile.h"

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

  std::string title("#splitline{#lower[.1]{#scale[.5]{");
  title += (roc / 18 % 2 == 0) ? "A" : "C";
  title += fmt::format("{:02d} ({:02d}) row: {:02d}, pad: {:03d}, globalpad: {:05d} (in roc)}}}{#scale[.5]{FEC: {:02d}, Chip: {:02d}, Chn: {:02d}, Value: {:.3f}}}",
                       roc % 18, roc, row, pad, channel, fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel(), binValue);

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

void utils::saveCanvas(TCanvas& c, std::string_view outDir, std::string_view types)
{
  const auto typesVec = tokenize(types, ",");
  for (const auto& type : typesVec) {
    c.SaveAs(fmt::format("{}/{}.{}", outDir, c.GetName(), type).data());
  }
}
