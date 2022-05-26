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

#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include "Framework/Logger.h"
#include "ZDCCalib/WaveformCalibParam.h"

using namespace o2::zdc;

void WaveformCalibChParam::print() const
{
  if (shape.size() > 0) {
    printf("Shape min at bin %d/%lu\n", ampMinID, shape.size());
  } else {
    printf("No data\n");
  }
}

void WaveformCalibParam::print() const
{
  for (int i = 0; i < NChannels; i++) {
    printf("%s ", channelName(i));
    channels[i].print();
  }
}

void WaveformCalibParam::assign(const WaveformCalibData& data)
{
  for (int isig = 0; isig < NChannels; isig++) {
    float entries = data.getEntries(isig);
    int peak = data.mPeak;
    if (entries > 0) {
      int ifirst = data.getFirstValid(isig);
      int ilast = data.getLastValid(isig);
      channels[isig].ampMinID = peak - ifirst;
      for (int ip = ifirst; ip <= ilast; ip++) {
        channels[isig].shape.push_back(data.mWave[isig].mData[ip] / entries);
      }
    }
  }
}

//______________________________________________________________________________
int WaveformCalibParam::saveDebugHistos(const std::string fn) const
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    auto& channel = channels[is];
    auto& shape = channel.shape;
    int nbx = shape.size();
    int iamin = channel.ampMinID;
    if (nbx > 0) {
      TString n = TString::Format("h%d", is);
      TString t = TString::Format("Waveform %d %s", is, ChannelNames[is].data());
      TH1F h(n, t, nbx, -0.5 - iamin, nbx - iamin - 0.5);
      for (int ibx = 0; ibx < nbx; ibx++) {
        h.SetBinContent(ibx + 1, shape[ibx]);
      }
      h.SetEntries(1);
      h.Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}
