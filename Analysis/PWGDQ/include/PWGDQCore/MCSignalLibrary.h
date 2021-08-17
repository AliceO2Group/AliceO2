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

MCSignal* o2::aod::dqmcsignals::GetMCSignal(const char* name)
{
  std::string nameStr = name;
  MCSignal* signal;
  // 1-prong signals
  if (!nameStr.compare("alicePrimary")) {
    MCProng prong(1);                                  // 1-generation prong
    prong.SetSourceBit(0, MCProng::kPhysicalPrimary);  // set source to be ALICE primary particles
    signal = new MCSignal(1, name, "ALICE primaries"); // define a signal with one prong
    signal->AddProng(prong);                           // add the previously defined prong to the signal
    return signal;
  }
  if (!nameStr.compare("electron")) {
    MCProng prong(1, {11}, {true}, {false}, {0}, {0}, {false});        // define 1-generation prong using the full constructor
    signal = new MCSignal(name, "Inclusive electrons", {prong}, {-1}); // define the signal using the full constructor
    return signal;
  }
  if (!nameStr.compare("muon")) {
    MCProng prong(1, {13}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "Inclusive muons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronNOTfromTransport")) {
    MCProng prong(1);
    prong.SetPDGcode(0, 11, true);
    prong.SetSourceBit(0, MCProng::kProducedInTransport, true); // exclude particles produces in transport
    signal = new MCSignal(name, "Electrons which are not produced during transport in detector", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronFromJpsi")) {
    MCProng prong(2, {11, 443}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "Electrons from jpsi decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronFromNonpromptJpsi")) {
    MCProng prong(3, {11, 443, 503}, {true, true, true}, {false, false, false}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    signal = new MCSignal(name, "Electrons from beauty jpsi decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronFromPromptJpsi")) {
    MCProng prong(3, {11, 443, 503}, {true, true, true}, {false, false, true}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    signal = new MCSignal(name, "Electrons from beauty jpsi decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("jpsi")) {
    MCProng prong(1, {443}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "Inclusive jpsi", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("nonPromptJpsi")) {
    MCProng prong(2, {443, 503}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "Non-prompt jpsi", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("promptJpsi")) {
    MCProng prong(2, {443, 503}, {true, true}, {false, true}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "Prompt jpsi (not from beauty)", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("anyBeautyHadron")) {
    MCProng prong(1, {503}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "All beauty hadrons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeauty")) {
    MCProng prong(2, {0, 503}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "Everything from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("everythingFromEverythingFromBeauty")) {
    MCProng prong(3, {0, 0, 503}, {true, true, true}, {false, false, false}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    signal = new MCSignal(name, "Everything from everything from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("allCharmHadrons")) {
    MCProng prong(1, {403}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "All charm hadrons", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("allCharmFromBeauty")) {
    MCProng prong(2, {403, 503}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "All charm hadrons from beauty", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("allPromptCharm")) {
    MCProng prong(2, {403, 503}, {true, true}, {false, true}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "All prompt charm hadrons (not from beauty)", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronFromPi0")) {
    MCProng prong(2, {11, 111}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "Electrons from pi0 decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("Pi0")) {
    MCProng prong(1, {111}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "Pi0", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("electronFromDs")) {
    MCProng prong(2, {11, 431}, {true, true}, {false, false}, {0, 0}, {0, 0}, {true, true});
    signal = new MCSignal(name, "Electrons from Ds decays", {prong}, {-1});
    return signal;
  }
  if (!nameStr.compare("dsMeson")) {
    MCProng prong(1, {431}, {true}, {false}, {0}, {0}, {true});
    signal = new MCSignal(name, "Ds mesons", {prong}, {-1});
    return signal;
  }

  // 2-prong signals
  if (!nameStr.compare("dielectron")) {
    MCProng prong(1, {11}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal("dielectron", "Electron pair", {prong, prong}, {-1, -1});
    return signal;
  }
  if (!nameStr.compare("dimuon")) {
    MCProng prong(1, {13}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal("dielectron", "Electron pair", {prong, prong}, {-1, -1});
    return signal;
  }
  if (!nameStr.compare("electronMuonPair")) {
    MCProng electron(1, {11}, {true}, {false}, {0}, {0}, {false});
    MCProng muon(1, {13}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal(name, "Electron-muon pair", {electron, muon}, {-1, -1});
    return signal;
  }
  if (!nameStr.compare("dielectronFromPC")) {
    MCProng prong(2, {11, 22}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal(name, "dielectron from a photon conversion", {prong, prong}, {1, 1});
    return signal;
  }
  if (!nameStr.compare("dielectronPCPi0")) {
    MCProng prong(3, {11, 22, 111}, {true, true, true}, {false, false, false}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    signal = new MCSignal(name, "dielectron from a photon conversion from a pi0", {prong, prong}, {1, 1});
    return signal;
  }
  if (!nameStr.compare("beautyPairs")) {
    MCProng prong(1, {503}, {true}, {false}, {0}, {0}, {false});
    signal = new MCSignal("beautyPairs", "Beauty hadron pair", {prong, prong}, {-1, -1});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeautyPairs")) {
    MCProng prong(2, {0, 503}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal("everythingFromBeautyPairs", "Everything from beauty hadrons pair", {prong, prong}, {-1, -1});
    return signal;
  }
  if (!nameStr.compare("everythingFromEverythingFromBeautyPairsCM")) {
    MCProng prong(3, {0, 0, 503}, {true, true, true}, {false, false, false}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    signal = new MCSignal("everythingFromEverythingFromBeautyPairs", "Everything from everything from beauty hadrons pair with common grand-mother", {prong, prong}, {2, 2});
    return signal;
  }
  if (!nameStr.compare("everythingFromBeautyANDeverythingFromEverythingFromBeautyPairs")) {
    MCProng prong1(3, {0, 0, 503}, {true, true, true}, {false, false, false}, {0, 0, 0}, {0, 0, 0}, {false, false, false});
    MCProng prong2(2, {0, 503}, {true, true}, {false, false}, {0, 0}, {0, 0}, {false, false});
    signal = new MCSignal("everythingFromBeautyANDeverythingFromEverythingFromBeautyPairs", "Everything beauty and everything from everything from beauty hadrons pair", {prong1, prong2}, {2, 1});
    return signal;
  }
  return nullptr;
}
