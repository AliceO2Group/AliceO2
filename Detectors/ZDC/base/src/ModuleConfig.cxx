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
  printf("Modules configuration: baselineFactor = %f\n", baselineFactor);
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
  int ib = 0, nb = 0;
  uint64_t one = 0x1;
  std::string blist;
  for (int i = 0; i < NWMap; i++) {
    uint64_t val = emptyMap[i];
    for (int j = 0; j < 64; j++) {
      if ((val & (one << j)) != 0) { // Empty bunch
        blist += (" " + std::to_string(ib));
        nb++;
      }
      ib++;
      if (ib == o2::constants::lhc::LHCMaxBunches) {
        break;
      }
    }
  }
  LOG(info) << "Bunch list for baseline calculation:" << (nb == 0 ? " EMPTY" : blist);
}

//______________________________________________________________________________
void ModuleConfig::resetMap()
{
  for (int i = 0; i < NWMap; i++) {
    emptyMap[i] = 0;
  }
}

//______________________________________________________________________________
void ModuleConfig::addBunch(int ibunch)
{
  if (ibunch < 0 || ibunch >= o2::constants::lhc::LHCMaxBunches) {
    LOG(fatal) << "ModuleConfig::addBunch out of range [0:3563] " << ibunch;
  }
  int iw = ibunch / 64;
  int is = ibunch % 64;
  uint64_t one = 0x1;
  emptyMap[iw] = emptyMap[iw] | (one << is);
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
    LOG(fatal) << "ZDC Module can have at most " << Module::MaxTriggChannels << " trigger channels";
  }
}

//______________________________________________________________________________
void Module::setChannel(int slot, int8_t chID, int16_t fID, bool read, bool trig, int tF, int tL, int tS, int tT)
{
  if (slot < 0 || slot >= MaxChannels || chID < 0 || chID > NChannels) {
    LOG(fatal) << "Improper module channel settings" << slot << ' ' << chID << ' ' << fID << ' ' << read << ' ' << trig
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
      LOG(fatal) << "Sum of Last and Shift trigger parameters exceed allowed range";
    }
    trigChannelConf[slot].id = chID;
    trigChannelConf[slot].first = tF;
    trigChannelConf[slot].last = tL;
    trigChannelConf[slot].shift = tS;
    trigChannelConf[slot].threshold = tT;
  }
}

//______________________________________________________________________________
uint32_t ModuleConfig::getTriggerMask() const
{
  uint32_t triggermask = 0;
  for (int im = 0; im < NModules; im++) {
    for (int ic = 0; ic < NChPerModule; ic++) {
      if (modules[im].trigChannel[ic]) {
        uint32_t tmask = 0x1 << (im * NChPerModule + ic);
        triggermask = triggermask | tmask;
      }
    }
  }
  return triggermask;
}

std::string ModuleConfig::getPrintTriggerMask() const
{
  std::string printTriggerMask{};
  for (int im = 0; im < NModules; im++) {
    if (im > 0) {
      printTriggerMask += " ";
    }
    printTriggerMask += std::to_string(im);
    printTriggerMask += "[";
    for (int ic = 0; ic < NChPerModule; ic++) {
      if (modules[im].trigChannel[ic]) {
        printTriggerMask += "T";
      } else {
        printTriggerMask += " ";
      }
    }
    printTriggerMask += "]";
  }
  return printTriggerMask;
}
