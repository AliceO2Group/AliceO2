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

#include "DataFormatsTPC/CalibdEdxCorrection.h"

#include <string_view>

// o2 includes
#include "DataFormatsTPC/Defs.h"
#include "CommonUtils/TreeStreamRedirector.h"

// root includes
#include "TFile.h"

using namespace o2::tpc;

void CalibdEdxCorrection::clear()
{
  for (auto& row : mParams) {
    for (auto& x : row) {
      x = 0.f;
    }
  }
  for (auto& x : mChi2) {
    x = 0.f;
  }
  mDims = -1;
}

void CalibdEdxCorrection::writeToFile(std::string_view fileName) const
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.data(), "recreate"));
  file->WriteObject(this, "CalibdEdxCorrection");
}

void CalibdEdxCorrection::loadFromFile(std::string_view fileName)
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.data()));
  auto tmp = file->Get<CalibdEdxCorrection>("CalibdEdxCorrection");
  if (tmp != nullptr) {
    *this = *tmp;
  }
}

void CalibdEdxCorrection::dumpToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  for (int sector = 0; sector < 2 * SECTORSPERSIDE; ++sector) {
    for (int roc = 0; roc < GEMSTACKSPERSECTOR; ++roc) {
      tpc::StackID stack{
        sector,
        static_cast<tpc::GEMstack>(roc)};

      std::vector<float> qMaxCorrOut;
      std::vector<float> qTotCorrOut;
      std::vector<float> tglOut;
      std::vector<float> snpOut;

      for (float tgl = 0; tgl < 2; tgl += 0.01) {
        for (float snp = 0; snp < 1; snp += 0.1) {
          qMaxCorrOut.emplace_back(getCorrection(stack, ChargeType::Max, tgl, snp));
          qTotCorrOut.emplace_back(getCorrection(stack, ChargeType::Tot, tgl, snp));
          tglOut.emplace_back(tgl);
          snpOut.emplace_back(snp);
        }
      }

      pcstream << "tree"
               << "qMaxCorr=" << qMaxCorrOut
               << "qTotCorr=" << qTotCorrOut
               << "tgl=" << tglOut
               << "snp=" << snpOut
               << "roc=" << roc
               << "sector=" << sector
               << "\n";
    }
  }
}
