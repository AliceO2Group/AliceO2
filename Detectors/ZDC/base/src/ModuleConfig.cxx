// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCBase/ModuleConfig.h"
#include "ZDCBase/ModuleConfig.h"
#include <FairLogger.h>

using namespace o2::zdc;

void Module::print() const
{
  printf("Module %d [ChID/LinkID R:T ]", id);
  for (int ic = 0; ic < MaxChannels; ic++) {
    printf("[%s{%2d}/L%02d %c:%c ]", channelName(channelID[ic]), channelID[ic], linkID[ic], readChannel[ic] ? '+' : '-', trigChannel[ic] ? '+' : '-');
  }
  printf("\n");
  printf("Trigger conf: ");
  for (int ic = 0; ic < MaxChannels; ic++) {
    if (trigChannel[ic]) {
      const auto& cnf = trigChannelConf[ic];
      printf("[%s: F:%2d L:%2d S:%2d T:%2d] ", channelName(channelID[ic]), cnf.first, cnf.last, cnf.shift, cnf.threshold);
    }
  }
  printf("\n");
}

void ModuleConfig::print() const
{
  printf("Modules configuration:\n");
  for (const auto& md : modules) {
    if (md.id >= 0) {
      md.print();
    }
  }
}

void ModuleConfig::check() const
{
  for (const auto& md : modules) {
    md.check();
  }
}

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

void Module::setChannel(int slot, int8_t chID, int16_t lID, bool read, bool trig, int tF, int tL, int tS, int tT)
{
  if (slot < 0 || slot >= MaxChannels || chID < 0 || chID > NChannels) {
    LOG(FATAL) << "Improper module channel settings" << slot << ' ' << chID << ' ' << lID << ' ' << read << ' ' << trig
               << ' ' << tF << ' ' << tL << ' ' << tS << ' ' << tT;
  }
  linkID[slot] = lID;
  channelID[slot] = chID;
  readChannel[slot] = read;
  trigChannel[slot] = trig;
  if (trig) {
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
