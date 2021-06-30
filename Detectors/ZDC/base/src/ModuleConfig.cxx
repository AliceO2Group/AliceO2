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

#include "ZDCBase/ModuleConfig.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

//______________________________________________________________________________
void Module::printCh() const
{
  printf("Module %d [ChID/FEEID R:T ]", id);
  for (int ic = 0; ic < MaxChannels; ic++) {
    printf("[%s{%2d}/L%02d %c:%c ]", channelName(channelID[ic]), channelID[ic], feeID[ic], readChannel[ic] ? 'R' : ' ', trigChannel[ic] ? 'T' : ' ');
  }
  printf("\n");
}

//______________________________________________________________________________
void Module::printTrig() const
{
  printf("Trigger conf %d: ", id);
  for (int ic = 0; ic < MaxChannels; ic++) {
    const auto& cnf = trigChannelConf[ic];
    if (trigChannel[ic]) {
      printf("[TRIG %s: F:%2d L:%2d S:%2d T:%2d] ", channelName(channelID[ic]), cnf.first, cnf.last, cnf.shift, cnf.threshold);
    } else if (cnf.shift > 0 && cnf.threshold > 0) {
      printf("[DISC %s: F:%2d L:%2d S:%2d T:%2d] ", channelName(channelID[ic]), cnf.first, cnf.last, cnf.shift, cnf.threshold);
    }
  }
  printf("\n");
}

//______________________________________________________________________________
void Module::print() const
{
  printCh();
  printTrig();
}

void ModuleConfig::print() const
{
  printf("Modules configuration:\n");
  for (const auto& md : modules) {
    if (md.id >= 0) {
      md.printCh();
    }
  }
  for (const auto& md : modules) {
    if (md.id >= 0) {
      md.printTrig();
    }
  }
}

//______________________________________________________________________________
void ModuleConfig::check() const
{
  for (const auto& md : modules) {
    md.check();
  }
}

//______________________________________________________________________________
void Module::check() const
{
  // make sure that the channel has <= 2 triggers
  int ntr = 0;
  for (int i = MaxChannels; i--;) {
    ntr += trigChannel[i];
  }
  if (ntr > Module::MaxTriggChannels) {
    print();
    LOG(FATAL) << "ZDC Module can have at most " << Module::MaxTriggChannels << " trigger channels";
  }
}

//______________________________________________________________________________
void Module::setChannel(int slot, int8_t chID, int16_t fID, bool read, bool trig, int tF, int tL, int tS, int tT)
{
  if (slot < 0 || slot >= MaxChannels || chID < 0 || chID > NChannels) {
    LOG(FATAL) << "Improper module channel settings" << slot << ' ' << chID << ' ' << fID << ' ' << read << ' ' << trig
               << ' ' << tF << ' ' << tL << ' ' << tS << ' ' << tT;
  }
  feeID[slot] = fID;
  channelID[slot] = chID;
  readChannel[slot] = read;
  // In the 2020 firmware implementation, autotrigger bits are computed for each channel
  // Therefore we put the trig flag just for the triggering channels
  // Discriminator parameters are stored for all modules
  trigChannel[slot] = trig;
  if (tS > 0 && tT > 0) {
    if (tL + tS + 1 >= NTimeBinsPerBC) {
      LOG(FATAL) << "Sum of Last and Shift trigger parameters exceed allowed range";
    }
    trigChannelConf[slot].id = chID;
    trigChannelConf[slot].first = tF;
    trigChannelConf[slot].last = tL;
    trigChannelConf[slot].shift = tS;
    trigChannelConf[slot].threshold = tT;
  }
}
