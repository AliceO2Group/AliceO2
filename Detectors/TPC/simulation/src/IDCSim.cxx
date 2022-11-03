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

#include "TPCSimulation/IDCSim.h"
#include "DataFormatsTPC/Digit.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TFile.h"
#include "TPCBase/Mapper.h"
#include <fmt/format.h>
#include "Framework/Logger.h"
#include "TPCBase/Painter.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TKey.h"

void o2::tpc::IDCSim::integrateDigitsForOneTF(const gsl::span<const o2::tpc::Digit>& digits)
{
  resetIDCs();

  // loop over digits from one sector for ALL Time Frames
  const unsigned int switchAfterTB = getLastTimeBinForSwitch();

  if (mAddInterval) {
    // decrease the size of the vector if the last integration intervall is empty
    if (switchAfterTB == (mIntegrationIntervalsPerTF - 1) * mTimeStampsPerIntegrationInterval) {
      for (unsigned int ireg = 0; ireg < Mapper::NREGIONS; ++ireg) {
        mIDCs[mBufferIndex][ireg].resize(mMaxIDCs[ireg] - Mapper::PADSPERREGION[ireg]);
      }
    } else {
      for (auto& idcs : mIDCs[mBufferIndex]) {
        idcs.resize(idcs.capacity());
      }
    }
  }

  for (const auto& digit : digits) {
    const o2::tpc::CRU cru(digit.getCRU());
    const unsigned int region = cru.region();
    const int timeStamp = digit.getTimeStamp();
    if (timeStamp < switchAfterTB) {
      const unsigned int indexPad = getIndex(timeStamp, region, digit.getRow(), digit.getPad());
      mIDCs[mBufferIndex][region][indexPad] += digit.getChargeFloat();
    } else {
      const unsigned int indexPad = getIndex(timeStamp - switchAfterTB, region, digit.getRow(), digit.getPad());
      mIDCs[!mBufferIndex][region][indexPad] += digit.getChargeFloat();
    }
  }

  // normalize IDCs as they are normalized for the real data
  const float norm = 1. / float(mTimeStampsPerIntegrationInterval);
  for (auto& idc : mIDCs[mBufferIndex]) {
    std::transform(idc.begin(), idc.end(), idc.begin(), [norm](float& val) { return val * norm; });
  }

  mBufferIndex = !mBufferIndex; // switch buffer index
  setNewOffset();               // set offset
}

unsigned int o2::tpc::IDCSim::getLastTimeBinForSwitch() const
{
  const int totaloffs = mTimeBinsOff + static_cast<int>(mTimeStampsRemainder);
  return (totaloffs >= mTimeStampsPerIntegrationInterval) ? mIntegrationIntervalsPerTF * mTimeStampsPerIntegrationInterval - mTimeBinsOff : (mIntegrationIntervalsPerTF - mAddInterval) * mTimeStampsPerIntegrationInterval - mTimeBinsOff;
}

void o2::tpc::IDCSim::setNewOffset()
{
  const int totaloffs = mTimeBinsOff + static_cast<int>(mTimeStampsRemainder);
  mTimeBinsOff = (totaloffs >= mTimeStampsPerIntegrationInterval) ? (totaloffs - static_cast<int>(mTimeStampsPerIntegrationInterval)) : totaloffs;
}

/// set all IDC values to 0
void o2::tpc::IDCSim::resetIDCs()
{
  for (auto& idcs : mIDCs[!mBufferIndex]) {
    std::fill(idcs.begin(), idcs.end(), 0);
  }
}

void o2::tpc::IDCSim::dumpIDCs(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::IDCSim::createDebugTree(const char* nameTree) const
{
  o2::utils::TreeStreamRedirector pcstream(nameTree, "RECREATE");
  pcstream.GetFile()->cd();
  createDebugTree(*this, pcstream);
  pcstream.Close();
}

void o2::tpc::IDCSim::createDebugTreeForAllSectors(const char* nameTree, const char* filename)
{
  o2::utils::TreeStreamRedirector pcstream(nameTree, "RECREATE");
  pcstream.GetFile()->cd();

  TFile fInp(filename, "READ");
  for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
    const auto key = dynamic_cast<TKey*>(keyAsObj);
    LOGP(info, "Key name: {} Type: {}", key->GetName(), key->GetClassName());

    if (std::strcmp(o2::tpc::IDCSim::Class()->GetName(), key->GetClassName()) != 0) {
      LOGP(info, "skipping object. wrong class.");
      continue;
    }

    IDCSim* idcsim = (IDCSim*)fInp.Get(key->GetName());
    createDebugTree(*idcsim, pcstream);
    delete idcsim;
  }
  pcstream.Close();
}

void o2::tpc::IDCSim::createDebugTree(const IDCSim& idcsim, o2::utils::TreeStreamRedirector& pcstream)
{
  const Mapper& mapper = Mapper::instance();
  const unsigned int sector = idcsim.getSector();
  unsigned int cru = sector * Mapper::NREGIONS;

  // loop over data from regions
  for (const auto& idcs : idcsim.get()) {
    int sectorTmp = sector;
    const o2::tpc::CRU cruTmp(cru);
    unsigned int region = cruTmp.region();
    const unsigned long padsPerCRU = Mapper::PADSPERREGION[region];
    std::vector<int> vRow(padsPerCRU);
    std::vector<int> vPad(padsPerCRU);
    std::vector<float> vXPos(padsPerCRU);
    std::vector<float> vYPos(padsPerCRU);
    std::vector<float> vGlobalXPos(padsPerCRU);
    std::vector<float> vGlobalYPos(padsPerCRU);
    std::vector<float> idcsPerTimeBin(padsPerCRU); // idcs for one time bin

    for (unsigned int iPad = 0; iPad < padsPerCRU; ++iPad) {
      const GlobalPadNumber globalNum = Mapper::GLOBALPADOFFSET[region] + iPad;
      const auto& padPosLocal = mapper.padPos(globalNum);
      vRow[iPad] = padPosLocal.getRow();
      vPad[iPad] = padPosLocal.getPad();
      vXPos[iPad] = mapper.getPadCentre(padPosLocal).X();
      vYPos[iPad] = mapper.getPadCentre(padPosLocal).Y();
      const GlobalPosition2D globalPos = mapper.LocalToGlobal(LocalPosition2D(vXPos[iPad], vYPos[iPad]), cruTmp.sector());
      vGlobalXPos[iPad] = globalPos.X();
      vGlobalYPos[iPad] = globalPos.Y();
    }

    for (unsigned int integrationInterval = 0; integrationInterval < idcsim.getNIntegrationIntervalsPerTF(); ++integrationInterval) {
      for (unsigned int iPad = 0; iPad < padsPerCRU; ++iPad) {
        idcsPerTimeBin[iPad] = (idcs)[iPad + integrationInterval * Mapper::PADSPERREGION[region]];
      }

      pcstream << "tree"
               << "cru=" << cru
               << "sector=" << sectorTmp
               << "region=" << region
               << "integrationInterval=" << integrationInterval
               << "IDC.=" << idcsPerTimeBin
               << "pad.=" << vPad
               << "row.=" << vRow
               << "lx.=" << vXPos
               << "ly.=" << vYPos
               << "gx.=" << vGlobalXPos
               << "gy.=" << vGlobalYPos
               << "\n";
    }
    ++cru;
  }
}

void o2::tpc::IDCSim::drawIDCs(const unsigned int integrationInterval, const std::string filename) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{IDC}");
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);

  TLatex lat;
  lat.SetTextFont(63);
  lat.SetTextSize(2);

  poly->Draw("colz");
  for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
    for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
      for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
        const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
        const auto coordinate = coords[padNum];
        const float yPos = -0.5 * (coordinate.yVals[0] + coordinate.yVals[2]); // local coordinate system is mirrored
        const float xPos = 0.5 * (coordinate.xVals[0] + coordinate.xVals[2]);
        const unsigned int indexIDC = integrationInterval * Mapper::PADSPERREGION[region] + Mapper::OFFSETCRULOCAL[region][irow] + ipad;
        const float idc = mIDCs[!mBufferIndex][region][indexIDC];
        poly->Fill(xPos, yPos, idc);
        lat.SetTextAlign(12);
        lat.DrawLatex(xPos, yPos, Form("%i", ipad));
      }
    }
  }

  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}
