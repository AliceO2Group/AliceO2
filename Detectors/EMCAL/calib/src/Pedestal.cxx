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

#include "EMCALBase/Geometry.h"
#include "EMCALCalib/CalibContainerErrors.h"
#include "EMCALCalib/Pedestal.h"

#include <fairlogger/Logger.h>

#include <TH1.h>
#include <TH2.h>

#include <iostream>

#include <gsl/span>

using namespace o2::emcal;

void Pedestal::addPedestalValue(unsigned short cellID, short pedestal, bool isLowGain, bool isLEDMON)
{
  if (cellID >= mPedestalValuesHG.size()) {
    throw CalibContainerIndexException(cellID);
  }
  if (isLEDMON) {
    if (isLowGain) {
      mPedestalValuesLEDMONLG[cellID] = pedestal;
    } else {
      mPedestalValuesLEDMONHG[cellID] = pedestal;
    }
  } else {
    if (isLowGain) {
      mPedestalValuesLG[cellID] = pedestal;
    } else {
      mPedestalValuesHG[cellID] = pedestal;
    }
  }
}

short Pedestal::getPedestalValue(unsigned short cellID, bool isLowGain, bool isLEDMON) const
{
  if (cellID >= mPedestalValuesHG.size()) {
    throw CalibContainerIndexException(cellID);
  }
  if (isLEDMON) {
    if (isLowGain) {
      return mPedestalValuesLEDMONLG[cellID];
    } else {
      return mPedestalValuesLEDMONHG[cellID];
    }
  } else {
    if (isLowGain) {
      return mPedestalValuesLG[cellID];
    } else {
      return mPedestalValuesHG[cellID];
    }
  }
}

TH1* Pedestal::getHistogramRepresentation(bool isLowGain, bool isLEDMON) const
{
  gsl::span<const short> data;
  std::string histname, histtitle;
  if (isLEDMON) {
    if (isLowGain) {
      histname = "PedestalLG_LEDMON";
      histtitle = "LEDMON Pedestal values Params low Gain";
      data = gsl::span<const short>(mPedestalValuesLEDMONLG);
    } else {
      histname = "PedestalHG_LEDMON";
      histtitle = "LEDMON Pedestal values Params high Gain";
      data = gsl::span<const short>(mPedestalValuesLEDMONHG);
    }
  } else {
    if (isLowGain) {
      histname = "PedestalLG";
      histtitle = "Pedestal values Params low Gain";
      data = gsl::span<const short>(mPedestalValuesLG);
    } else {
      histname = "PedestalHG";
      histtitle = "Pedestal values Params high Gain";
      data = gsl::span<const short>(mPedestalValuesHG);
    }
  }

  auto hist = new TH1S(histname.data(), histtitle.data(), data.size(), -0.5, data.size() - 0.5);
  hist->SetDirectory(nullptr);
  for (std::size_t icell{0}; icell < data.size(); ++icell) {
    hist->SetBinContent(icell + 1, data[icell]);
  }
  return hist;
}

TH2* Pedestal::getHistogramRepresentation2D(bool isLowGain, bool isLEDMON) const
{
  gsl::span<const short> data;
  std::string histname, histtitle;
  if (isLEDMON) {
    if (isLowGain) {
      histname = "PedestalLG_LEDMON";
      histtitle = "LEDMON Pedestal values Params low Gain";
      data = gsl::span<const short>(mPedestalValuesLEDMONLG);
    } else {
      histname = "PedestalHG_LEDMON";
      histtitle = "LEDMON Pedestal values Params high Gain";
      data = gsl::span<const short>(mPedestalValuesLEDMONHG);
    }
  } else {
    if (isLowGain) {
      histname = "PedestalLG";
      histtitle = "Pedestal values Params low Gain";
      data = gsl::span<const short>(mPedestalValuesLG);
    } else {
      histname = "PedestalHG";
      histtitle = "Pedestal values Params high Gain";
      data = gsl::span<const short>(mPedestalValuesHG);
    }
  }

  const int MAXROWS = isLEDMON ? 10 : 208,
            MAXCOLS = isLEDMON ? 48 : 96;

  auto hist = new TH2S(histname.data(), histtitle.data(), MAXCOLS, -0.5, double(MAXCOLS) - 0.5, MAXROWS, -0.5, double(MAXROWS) - 0.5);
  hist->SetDirectory(nullptr);
  try {
    auto geo = Geometry::GetInstance();
    for (size_t ichan = 0; ichan < data.size(); ichan++) {
      if (isLEDMON) {
        int col = ichan % 48,
            row = ichan / 48;
        hist->Fill(col, row, data[ichan]);
      } else {
        auto position = geo->GlobalRowColFromIndex(ichan);
        hist->Fill(std::get<1>(position), std::get<0>(position), data[ichan]);
      }
    }
  } catch (o2::emcal::GeometryNotInitializedException& e) {
    LOG(error) << "Geometry needs to be initialized";
  }
  return hist;
}

bool Pedestal::operator==(const Pedestal& other) const
{
  return mPedestalValuesHG == other.mPedestalValuesHG && mPedestalValuesLG == other.mPedestalValuesLG && mPedestalValuesLEDMONHG == other.mPedestalValuesLEDMONHG && mPedestalValuesLEDMONLG == other.mPedestalValuesLEDMONLG;
}
