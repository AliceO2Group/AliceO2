// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
#include "PWGDQCore/MCProng.h"
#include "PWGDQCore/MCSignal.h"

namespace o2::aod
{
namespace dqmcsignals
{
MCSignal* GetMCSignal(const char* signalName);
}
} // namespace o2::aod

MCSignal* o2::aod::dqmcsignals::GetMCSignal(const char* name) {
  std::string nameStr = name;
  MCSignal* signal;
  if (!nameStr.compare("alicePrimary")) {
    MCProng prong(1,{0},{false},{false},{MCProng::kPhysicalPrimary},{0},{true});
    signal = new MCSignal("alicePrimary", "ALICE primaries", {prong},{-1});
    return signal;
  }
  if (!nameStr.compare("allBeautyHadrons")) {
    MCProng prong(1,{503},{true},{false},{0},{0},{false});
    signal = new MCSignal("allBeautyHadrons", "All beauty hadrons", {prong},{-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeauty")) {
    MCProng prong(2,{0,503},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("everythingFromBeauty", "Everything from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromEverythingFromBeauty")) {
    MCProng prong(3,{0,0,503},{true,true,true},{false,false,false},{0,0,0},{0,0,0},{false,false,false});
    signal = new MCSignal("everythingFromEverythingFromBeauty", "Everything from everything from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("allCharmHadrons")) {
    MCProng prong(1,{403},{true},{false},{0},{0},{false});
    signal = new MCSignal("allCharmHadrons", "All charm hadrons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("allCharmFromBeauty")) {
    MCProng prong(2,{403,503},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("allCharmFromBeauty", "All charm hadrons from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronPi0")) {
    MCProng prong(2,{11,111},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("electronPi0", "Electrons from pi0 decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("Pi0")) {
    MCProng prong(1,{111},{true},{false},{0},{0},{false});
    signal = new MCSignal("Pi0", "Pi0", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("inclusiveElectron")) {
    MCProng prong(1,{11},{true},{false},{0},{0},{false});
    signal = new MCSignal("inclusiveElectron", "Inclusive electrons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronsNOTfromTransport")) {
    MCProng prong(1);
    prong.SetPDGcode(0,11,true);
    prong.SetSourceBit(0,MCProng::kProducedInTransport,true);   // exclude particles produces in transport
    signal = new MCSignal("electronsNOTfromTransport", "Inclusive electrons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("jpsiElectron")) {
    MCProng prong(2,{11,443},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("jpsiElectron", "Electrons from inclusive jpsi decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("jpsiBeautyElectron")) {
    MCProng prong(3,{11,443,502},{true,true,true},{false,false,false},{0,0,0},{0,0,0},{false,false,false});
    signal = new MCSignal("jpsiBeautyElectron", "Electrons from beauty jpsi decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("dsElectron")) {
    MCProng prong(2,{11,431},{true,true},{false,false},{0,0},{0,0},{true,true});
    signal = new MCSignal("dsElectron", "Electrons from Ds decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("dsMeson")) {
    MCProng prong(1,{431},{true},{false},{0},{0},{true});
    signal = new MCSignal("dsMeson", "Ds mesons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("pionEta")) {
    MCProng prong(2,{211,221},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("pionEta", "Pions from eta decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("dielectronPC")) {
    MCProng prong(2,{11,22},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("dielectronPC", "dielectron from photon conversion", {prong,prong}, {-1,-1});
    return signal;
  }
  if (!nameStr.compare("dielectronPCPi0")) {
    MCProng prong(3,{11,22,111},{true,true,true},{false,false,false},{0,0,0},{0,0,0},{false,false,false});
    signal = new MCSignal("dielectronPC", "dielectron from photon conversion", {prong,prong}, {-1,-1});
    return signal;
  }
  if (!nameStr.compare("dielectron")) {
    MCProng prong(1,{11},{true},{false},{0},{0},{false});
    signal = new MCSignal("dielectron", "Electron pair", {prong,prong}, {-1,-1});
    return signal;
  }
  if (!nameStr.compare("beautyPairs")) {
    MCProng prong(1,{503},{true},{false},{0},{0},{false});
    signal = new MCSignal("beautyPairs", "Beauty hadron pair", {prong,prong}, {-1,-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeautyPairs")) {
    MCProng prong(2,{0,503},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("everythingFromBeautyPairs", "Everything from beauty hadrons pair", {prong,prong}, {-1,-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromEverythingFromBeautyPairs")) {
    MCProng prong(3,{0,0,503},{true,true,true},{false,false,false},{0,0,0},{0,0,0},{false,false,false});
    signal = new MCSignal("everythingFromEverythingFromBeautyPairs", "Everything from everything from beauty hadrons pair", {prong,prong}, {2,2});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeautyANDeverythingFromEverythingFromBeautyPairs")) {
    MCProng prong1(3,{0,0,503},{true,true,true},{false,false,false},{0,0,0},{0,0,0},{false,false,false});
    MCProng prong2(2,{0,503},{true,true},{false,false},{0,0},{0,0},{false,false});
    signal = new MCSignal("everythingFromBeautyANDeverythingFromEverythingFromBeautyPairs", "Everything beauty and everything from everything from beauty hadrons pair", {prong1,prong2}, {2,1});
    return signal;
  }
  return nullptr;
}
