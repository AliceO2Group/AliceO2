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

#include <Steer/O2MCApplication.h>
#include <fairmq/Channel.h>
#include <fairmq/Message.h>
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TMessage.h>
#include <sstream>
#include <SimConfig/SimConfig.h>
#include <DetectorsBase/Detector.h>
#include "DetectorsBase/Aligner.h"
#include "DetectorsBase/MaterialManager.h"
#include <CommonUtils/ShmManager.h>
#include <cassert>
#include <SimulationDataFormat/MCEventHeader.h>
#include <TGeoManager.h>
#include <fstream>
#include <FairVolume.h>
#include <CommonUtils/NameConf.h>
#include <CommonUtils/ConfigurationMacroHelper.h>
#include "SimConfig/SimUserDecay.h"
#include <filesystem>
#include <CommonUtils/FileSystemUtils.h>
#include "SimConfig/GlobalProcessCutSimParam.h"

namespace o2
{
namespace steer
{
// helper function to send trivial data
template <typename T>
void TypedVectorAttach(const char* name, fair::mq::Channel& channel, fair::mq::Parts& parts)
{
  static auto mgr = FairRootManager::Instance();
  auto vector = mgr->InitObjectAs<const std::vector<T>*>(name);
  if (vector) {
    auto buffer = (char*)&(*vector)[0];
    auto buffersize = vector->size() * sizeof(T);
    fair::mq::MessagePtr message(channel.NewMessage(
      buffer, buffersize,
      [](void* data, void* hint) {}, buffer));
    parts.AddPart(std::move(message));
  }
}

void O2MCApplicationBase::Stepping()
{
  mStepCounter++;

  // check the max time of flight condition
  const auto tof = fMC->TrackTime();
  auto& params = o2::GlobalProcessCutSimParam::Instance();
  if (tof > params.TOFMAX) {
    fMC->StopTrack();
    return;
  }

  mLongestTrackTime = std::max((double)mLongestTrackTime, tof);

  if (mCutParams.stepFiltering) {
    // we can kill tracks here based on our
    // custom detector specificities

    // Note that this is done in addition to the generic
    // R + Z-cut mechanism at VMC level.

    float x, y, z;
    fMC->TrackPosition(x, y, z);

    // this function is implementing a basic z-dependent R cut
    // can be generalized later on
    auto outOfR = [x, y, this](float z) {
      // for the moment for cases when we have ZDC enabled
      if (std::abs(z) > mCutParams.tunnelZ) {
        if ((x * x + y * y) > mCutParams.maxRTrackingZDC * mCutParams.maxRTrackingZDC) {
          return true;
        }
      }
      return false;
    };

    if (z > mCutParams.ZmaxA ||
        -z > mCutParams.ZmaxC ||
        outOfR(z)) {
      fMC->StopTrack();
      return;
    }
  }

  if (mCutParams.stepTrackRefHook) {
    mTrackRefFcn(fMC);
  }

  // dispatch now to stepping function in FairRoot
  FairMCApplication::Stepping();
}

void O2MCApplicationBase::PreTrack()
{
  // dispatch first to function in FairRoot
  FairMCApplication::PreTrack();
}

void O2MCApplicationBase::ConstructGeometry()
{
  // fill the mapping
  mModIdToName.clear();
  o2::detectors::DetID::mask_t dmask{};
  for (int i = 0; i < fModules->GetEntries(); ++i) {
    auto mod = static_cast<FairModule*>(fModules->At(i));
    if (mod) {
      mModIdToName[mod->GetModId()] = mod->GetName();
      int did = o2::detectors::DetID::nameToID(mod->GetName());
      if (did >= 0) {
        dmask |= o2::detectors::DetID::getMask(did);
      }
    }
  }
  gGeoManager->SetUniqueID(dmask.to_ulong());
  FairMCApplication::ConstructGeometry();

  std::ofstream voltomodulefile("MCStepLoggerVolMap.dat");
  // construct the volume name to module name mapping useful for StepAnalysis
  auto vollist = gGeoManager->GetListOfVolumes();
  for (int i = 0; i < vollist->GetEntries(); ++i) {
    auto vol = static_cast<TGeoVolume*>(vollist->At(i));
    auto iter = fModVolMap.find(vol->GetNumber());
    voltomodulefile << vol->GetName() << ":" << mModIdToName[iter->second] << "\n";
  }
}

void O2MCApplicationBase::InitGeometry()
{
  // load special cuts which might be given from the outside first.
  auto& matMgr = o2::base::MaterialManager::Instance();
  matMgr.loadCutsAndProcessesFromJSON(o2::base::MaterialManager::ESpecial::kTRUE);
  matMgr.SetLowEnergyNeutronTransport(mCutParams.lowneut);
  // During the following, FairModule::SetSpecialPhysicsCuts will be called for each module
  FairMCApplication::InitGeometry();
  matMgr.writeCutsAndProcessesToJSON();
  // now the sensitive volumes are set up in fVolMap and we can query them
  for (auto e : fVolMap) {
    // since fVolMap contains multiple entries (if multiple copies), this may
    // write to the same entry multiple times
    mSensitiveVolumes[e.first] = e.second->GetName();
  }
  std::ofstream sensvolfile("MCStepLoggerSenVol.dat");
  for (auto e : mSensitiveVolumes) {
    sensvolfile << e.first << ":" << e.second << "\n";
  }
}

bool O2MCApplicationBase::MisalignGeometry()
{
  for (auto det : listDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->addAlignableVolumes();
    }
  }

  // we stream out both unaligned geometry (to allow for
  // dynamic post-alignment) as well as the aligned version
  // which can be used by digitization etc. immediately
  auto& confref = o2::conf::SimConfig::Instance();
  auto geomfile = o2::base::NameConf::getGeomFileName(confref.getOutPrefix());
  // since in general the geometry is a CCDB object, it must be exported under the standard name
  gGeoManager->SetName(std::string(o2::base::NameConf::CCDBOBJECT).c_str());
  gGeoManager->Export(geomfile.c_str());

  // apply alignment for included detectors AFTER exporting ideal geometry
  auto& aligner = o2::base::Aligner::Instance();
  aligner.applyAlignment(confref.getTimestamp());

  // export aligned geometry into different file
  auto alignedgeomfile = o2::base::NameConf::getAlignedGeomFileName(confref.getOutPrefix());
  gGeoManager->Export(alignedgeomfile.c_str());

  // return original return value of misalignment procedure
  return true;
}

void O2MCApplicationBase::finishEventCommon()
{
  LOG(info) << "This event/chunk did " << mStepCounter << " steps";
  LOG(info) << "Longest track time is " << mLongestTrackTime;

  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  header->getMCEventStats().setNSteps(mStepCounter);
  header->setDetId2HitBitLUT(o2::base::Detector::getDetId2HitBitIndex());

  static_cast<o2::data::Stack*>(GetStack())->updateEventStats();
}

void O2MCApplicationBase::FinishEvent()
{
  finishEventCommon();

  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  auto& confref = o2::conf::SimConfig::Instance();

  if (confref.isFilterOutNoHitEvents() && header->getMCEventStats().getNHits() == 0) {
    LOG(info) << "Discarding current event due to no hits";
    SetSaveCurrentEvent(false);
  }

  // dispatch to function in FairRoot
  FairMCApplication::FinishEvent();
}

void O2MCApplicationBase::BeginEvent()
{
  // dispatch first to function in FairRoot
  FairMCApplication::BeginEvent();

  // register event header with our stack
  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  static_cast<o2::data::Stack*>(GetStack())->setMCEventStats(&header->getMCEventStats());

  mStepCounter = 0;
  mLongestTrackTime = 0;
}

void addSpecialParticles()
{
  //
  // Add particles needed for ALICE (not present in Geant3 or Geant4)
  // Code ported 1-1 from AliRoot
  //

  LOG(info) << "Adding custom particles to VMC";

  //Hypertriton
  TVirtualMC::GetMC()->DefineParticle(1010010030, "HyperTriton", kPTHadron, 2.99131, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);
  //Anti-Hypertriton
  TVirtualMC::GetMC()->DefineParticle(-1010010030, "AntiHyperTriton", kPTHadron, 2.99131, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);

  //Hyper hydrogen 4 ground state
  TVirtualMC::GetMC()->DefineParticle(1010010040, "Hyperhydrog4", kPTHadron, 3.9226, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  //Anti-Hyper hydrogen 4 ground state
  TVirtualMC::GetMC()->DefineParticle(-1010010040, "AntiHyperhydrog4", kPTHadron, 3.9226, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  //Hyper hydrogen 4 excited state
  TVirtualMC::GetMC()->DefineParticle(1010010041, "Hyperhydrog4*", kPTHadron, 3.9237, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  //Anti-Hyper hydrogen 4 excited state
  TVirtualMC::GetMC()->DefineParticle(-1010010041, "AntiHyperhydrog4*", kPTHadron, 3.9237, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  //Hyper helium 4 ground state
  TVirtualMC::GetMC()->DefineParticle(1010020040, "Hyperhelium4", kPTHadron, 3.9217, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  //Anti-Hyper helium 4 ground state
  TVirtualMC::GetMC()->DefineParticle(-1010020040, "AntiHyperhelium4", kPTHadron, 3.9217, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  //Hyper helium 4 excited state
  TVirtualMC::GetMC()->DefineParticle(1010020041, "Hyperhelium4*", kPTHadron, 3.9231, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  //Anti-Hyper helium 4 excited state
  TVirtualMC::GetMC()->DefineParticle(-1010020041, "AntiHyperhelium4*", kPTHadron, 3.9231, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  // Lithium 4 ground state
  TVirtualMC::GetMC()->DefineParticle(1000030040, "Lithium4", kPTHadron, 3.7513, 3.0, 9.1e-23, "Ion", 0.003, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  // Anti-Lithium 4 ground state
  TVirtualMC::GetMC()->DefineParticle(-1000030040, "AntiLithium4", kPTHadron, 3.7513, 3.0, 9.1e-23, "Ion", 0.003, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  //Hyper helium 5
  TVirtualMC::GetMC()->DefineParticle(1010020050, "Hyperhelium5", kPTHadron, 4.841, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 5, kFALSE);
  //Anti-Hyper helium 5
  TVirtualMC::GetMC()->DefineParticle(-1010020050, "AntiHyperhelium5", kPTHadron, 4.841, 2.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 5, kFALSE);

  //Double Hyper hydrogen 4
  TVirtualMC::GetMC()->DefineParticle(1020010040, "DoubleHyperhydrogen4", kPTHadron, 4.106, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);
  //Double Anti-Hyper hydrogen 4
  TVirtualMC::GetMC()->DefineParticle(-1020010040, "DoubleAntiHyperhydrogen4", kPTHadron, 4.106, 1.0, 2.632e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 4, kFALSE);

  //Lambda-Neutron
  TVirtualMC::GetMC()->DefineParticle(1010000020, "LambdaNeutron", kPTNeutron, 2.054, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Lambda-Neutron
  TVirtualMC::GetMC()->DefineParticle(-1010000020, "AntiLambdaNeutron", kPTNeutron, 2.054, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //H-Dibaryon
  TVirtualMC::GetMC()->DefineParticle(1020000020, "Hdibaryon", kPTNeutron, 2.23, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-H-Dibaryon
  TVirtualMC::GetMC()->DefineParticle(-1020000020, "AntiHdibaryon", kPTNeutron, 2.23, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Xi-Proton
  TVirtualMC::GetMC()->DefineParticle(1020010020, "Xi0Proton", kPTHadron, 2.248, 1.0, 1.333e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Xi-Proton
  TVirtualMC::GetMC()->DefineParticle(-1020010020, "AntiXi0Proton", kPTHadron, 2.248, 1.0, 1.333e-10, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Lambda-Neutron-Neutron
  TVirtualMC::GetMC()->DefineParticle(1010000030, "LambdaNeutronNeutron", kPTNeutron, 2.99, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);

  //Anti-Lambda-Neutron-Neutron
  TVirtualMC::GetMC()->DefineParticle(-1010000030, "AntiLambdaNeutronNeutron", kPTNeutron, 2.99, 0.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);

  //Omega-Proton
  TVirtualMC::GetMC()->DefineParticle(1030000020, "OmegaProton", kPTNeutron, 2.592, 0.0, 2.632e-10, "Hadron", 0.0, 2, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Omega-Proton
  TVirtualMC::GetMC()->DefineParticle(-1030000020, "AntiOmegaProton", kPTNeutron, 2.592, 0.0, 2.632e-10, "Hadron", 0.0, 2, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Omega-Neutron
  TVirtualMC::GetMC()->DefineParticle(1030010020, "OmegaNeutron", kPTHadron, 2.472, 1.0, 2.190e-22, "Hadron", 0.0, 2, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Omega-Neutron
  TVirtualMC::GetMC()->DefineParticle(-1030010020, "AntiOmegaNeutron", kPTHadron, 2.472, 1.0, 2.190e-22, "Hadron", 0.0, 2, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Omega-Omega
  TVirtualMC::GetMC()->DefineParticle(1060020020, "OmegaOmega", kPTHadron, 3.229, 2.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Omega-Omega
  TVirtualMC::GetMC()->DefineParticle(-1060020020, "AntiOmegaOmega", kPTHadron, 3.229, 2.0, 2.632e-10, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Lambda(1405)-Proton
  TVirtualMC::GetMC()->DefineParticle(1010010021, "Lambda1405Proton", kPTHadron, 2.295, 1.0, 1.316e-23, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Lambda(1405)-Proton
  TVirtualMC::GetMC()->DefineParticle(-1010010021, "AntiLambda1405Proton", kPTHadron, 2.295, 1.0, 1.316e-23, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Lambda(1405)-Lambda(1405)
  TVirtualMC::GetMC()->DefineParticle(1020000021, "Lambda1405Lambda1405", kPTNeutron, 2.693, 0.0, 1.316e-23, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Anti-Lambda(1405)-Lambda(1405)
  TVirtualMC::GetMC()->DefineParticle(-1020000021, "AntiLambda1405Lambda1405", kPTNeutron, 2.693, 0.0, 1.316e-23, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //c-deuteron
  TVirtualMC::GetMC()->DefineParticle(2010010020, "CDeuteron", kPTHadron, 3.226, 1.0, 2.0e-13, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);
  //Anti-c-deuteron
  TVirtualMC::GetMC()->DefineParticle(-2010010020, "AntiCDeuteron", kPTHadron, 3.226, 1.0, 2.0e-13, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 3, kFALSE);

  //c-triton
  TVirtualMC::GetMC()->DefineParticle(2010010030, "CTriton", kPTHadron, 4.162, 1.0, 2.0e-13, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);
  //Anti-c-Hypertriton
  TVirtualMC::GetMC()->DefineParticle(-2010010030, "AntiCTriton", kPTHadron, 4.162, 1.0, 2.0e-13, "Ion", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kFALSE);

  //Resonances not in Generators
  // f0(980) assume 70 MeV as width (PDG: 40 to 100 MeV)
  TVirtualMC::GetMC()->DefineParticle(9010221, "f0_980", kPTNeutron, 0.98, 0.0, 9.403e-24, "Hadron", 7e-2, 0, 1, 1, 0, 0, 1, 0, 0, kTRUE);

  // f2(1270) (PDG: width = 185 MeV)
  TVirtualMC::GetMC()->DefineParticle(225, "f2_1270", kPTNeutron, 1.275, 0.0, 3.558e-24, "Hadron", 0.185, 4, 1, 1, 0, 0, 1, 0, 0, kTRUE);

  // Xi_0(1820)
  TVirtualMC::GetMC()->DefineParticle(123324, "Xi_0_1820", kPTNeutron, 1.8234, 0.0, 2.742550e-23, "Hadron", 0.024, 3, -1, 0, 1, 1, 0, 0, 1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-123324, "Xi_0_Bar_1820", kPTNeutron, 1.8234, 0.0, 2.742550e-23, "Hadron", 0.024, 3, -1, 0, 1, -1, 0, 0, -1, kTRUE);

  int xi_0_1820_mode[6][3] = {{0}};
  float xi_0_1820_ratio[6] = {100.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  xi_0_1820_mode[0][0] = 3122; // Lambda
  xi_0_1820_mode[0][1] = 310;  // K0s
  TVirtualMC::GetMC()->SetDecayMode(123324, xi_0_1820_ratio, xi_0_1820_mode);
  xi_0_1820_mode[0][0] = -3122; // Lambda-bar
  TVirtualMC::GetMC()->SetDecayMode(-123324, xi_0_1820_ratio, xi_0_1820_mode);

  // Xi-+(1820)
  TVirtualMC::GetMC()->DefineParticle(123314, "Xi_Minus_1820", kPTHadron, 1.8234, -1.0, 2.742550e-23, "Hadron", 0.024, 3, -1, 0, 1, -1, 0, 0, 1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-123314, "Xi_Plus_1820", kPTHadron, 1.8234, 1.0, 2.742550e-23, "Hadron", 0.024, 3, -1, 0, 1, 1, 0, 0, -1, kTRUE);

  int xi_charged_1820_mode[6][3] = {{0}};
  float xi_charged_1820_ratio[6] = {100.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  xi_charged_1820_mode[0][0] = 3122; // Lambda
  xi_charged_1820_mode[0][1] = -321; // K-
  TVirtualMC::GetMC()->SetDecayMode(123314, xi_charged_1820_ratio, xi_charged_1820_mode);
  xi_charged_1820_mode[0][0] = -3122; // Lambda-bar
  xi_charged_1820_mode[0][1] = 321;   // K+
  TVirtualMC::GetMC()->SetDecayMode(-123314, xi_charged_1820_ratio, xi_charged_1820_mode);

  // Ps - hidden strange (s-sbar) pentaquarks
  TVirtualMC::GetMC()->DefineParticle(9322134, "Ps_2100", kPTHadron, 2.1, 1.0, 1.6455e-23, "Hadron", 4.e-2, 3, -1, 0, 0, 0, 0, 0, 1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-9322134, "AntiPs_2100", kPTHadron, 2.1, -1.0, 1.6455e-23, "Hadron", 4.e-2, 3, -1, 0, 0, 0, 0, 0, -1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(9322136, "Ps_2500", kPTHadron, 2.5, 1.0, 1.6455e-23, "Hadron", 4.e-2, 5, 1, 0, 0, 0, 0, 0, 1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-9322136, "AntiPs_2500", kPTHadron, 2.5, -1.0, 1.6455e-23, "Hadron", 4.e-2, 5, 1, 0, 0, 0, 0, 0, -1, kTRUE);

  Int_t psmode[6][3] = {0};
  Float_t psratio[6] = {0.f};
  psratio[0] = 100.;

  psmode[0][0] = 333;  // phi
  psmode[0][1] = 2212; // proton
  TVirtualMC::GetMC()->SetDecayMode(9322134, psratio, psmode);
  TVirtualMC::GetMC()->SetDecayMode(9322136, psratio, psmode);

  psmode[0][1] = -2212; // anti-proton
  TVirtualMC::GetMC()->SetDecayMode(-9322134, psratio, psmode);
  TVirtualMC::GetMC()->SetDecayMode(-9322136, psratio, psmode);

  //Omega(2012)
  for (int j = 1; j < 6; j++) {
    psmode[j][0] = psmode[j][1] = 0;
    psratio[j] = 0.;
  }

  TVirtualMC::GetMC()->DefineParticle(3335, "Omega2012", kPTHadron, 2.012, -1.0, 1.0285e-22, "Hadron", 0.0064, 3, -1, 0, 0, 0, 0, 0, 1, kTRUE);
  psmode[0][0] = 3312; // Xi-
  psmode[0][1] = 310;  // K0S
  psratio[0] = 100.;
  TVirtualMC::GetMC()->SetDecayMode(3335, psratio, psmode);

  TVirtualMC::GetMC()->DefineParticle(-3335, "AntiOmega2012", kPTHadron, 2.012, 1.0, 1.0285e-22, "Hadron", 0.0064, 3, 1, 0, 0, 0, 0, 0, -1, kTRUE);
  psmode[0][0] = -3312; // anti-Xi+
  psmode[0][1] = 310;   // K0S
  psratio[0] = 100.;
  TVirtualMC::GetMC()->SetDecayMode(-3335, psratio, psmode);

  // d*(2380) - dibaryon resonance
  TVirtualMC::GetMC()->DefineParticle(900010020, "d*_2380", kPTHadron, 2.38, 1.0, 0.94e-23, "Ion", 0.07, 6, 1, 0, 0, 0, 0, 0, 2, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-900010020, "d*_2380_bar", kPTHadron, 2.38, -1.0, 0.94e-23, "Ion", 0.07, 6, 1, 0, 0, 0, 0, 0, -2, kTRUE);

  Int_t dstmode[6][3] = {0};
  Float_t dstratio[6] = {0.f};
  dstratio[0] = 100; // For now we implement only the mode of interest
  // d* -> d pi+ pi-
  dstmode[0][0] = 1000010020; // deuteron
  dstmode[0][1] = -211;       // negative pion
  dstmode[0][2] = 211;        // positive pion
  TVirtualMC::GetMC()->SetDecayMode(900010020, dstratio, dstmode);

  dstmode[0][0] = -1000010020; // anti-deuteron
  TVirtualMC::GetMC()->SetDecayMode(-900010020, dstratio, dstmode);

  // Heavy vector mesons
  // D*+
  TVirtualMC::GetMC()->DefineParticle(413, "D*+", kPTHadron, 2.0103, 1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // D*-
  TVirtualMC::GetMC()->DefineParticle(-413, "D*-", kPTHadron, 2.0103, -1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // D*0
  TVirtualMC::GetMC()->DefineParticle(423, "D*0", kPTHadron, 2.0007, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // D*0bar
  TVirtualMC::GetMC()->DefineParticle(-423, "D*0bar", kPTHadron, 2.0007, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // D*_s+
  TVirtualMC::GetMC()->DefineParticle(433, "D*_s+", kPTHadron, 2.1123, 1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // D*_s-
  TVirtualMC::GetMC()->DefineParticle(-433, "D*_s-", kPTHadron, 2.1123, -1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*0
  TVirtualMC::GetMC()->DefineParticle(513, "B*0", kPTHadron, 5.3251, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*0bar
  TVirtualMC::GetMC()->DefineParticle(-513, "B*0bar", kPTHadron, 5.3251, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*+
  TVirtualMC::GetMC()->DefineParticle(523, "B*+", kPTHadron, 5.3251, 1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*-
  TVirtualMC::GetMC()->DefineParticle(-523, "B*-", kPTHadron, 5.3251, -1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*_s0
  TVirtualMC::GetMC()->DefineParticle(533, "B*_s0", kPTHadron, 5.4128, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*_s0bar
  TVirtualMC::GetMC()->DefineParticle(-533, "B*_s0bar", kPTHadron, 5.4128, 0.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*_c+
  TVirtualMC::GetMC()->DefineParticle(543, "B*_c+", kPTHadron, 6.6020, 1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);
  // B*_c-
  TVirtualMC::GetMC()->DefineParticle(-543, "B*_c-", kPTHadron, 6.6020, -1.0, 0.0, "Hadron", 0.0, 1, -1, 0, 0, 0, 0, 0, 0, kTRUE);

  // Charm pentaquarks
  // Theta_c: isospin singlet with J=1/2+ (see https://arxiv.org/abs/hep-ph/0409121)
  TVirtualMC::GetMC()->DefineParticle(9422111, "Anti-Theta_c_3100", kPTHadron, 3.099, 0., 6.9e-21, "Hadron", 83.e-6, 1, 1, 0, 0, 0, 0, 0, -1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-9422111, "Theta_c_3100", kPTHadron, 3.099, 0., 6.9e-21, "Hadron", 83.e-6, 1, 1, 0, 0, 0, 0, 0, 1, kTRUE);

  for (int j = 1; j < 6; j++) {
    psmode[j][0] = psmode[j][1] = 0;
    psratio[j] = 0.;
  }
  psmode[0][0] = 413;   // D*+
  psmode[0][1] = -2212; // anti-p
  psratio[0] = 100.;
  TVirtualMC::GetMC()->SetDecayMode(9422111, psratio, psmode);
  psmode[0][0] = -413; // D*-
  psmode[0][1] = 2212; // p
  TVirtualMC::GetMC()->SetDecayMode(-9422111, psratio, psmode);

  // Define the 2- and 3-body phase space decay for the Hyper-Triton
  Int_t mode[6][3];
  Float_t bratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio[kz] = 0.;
    mode[kz][0] = 0;
    mode[kz][1] = 0;
    mode[kz][2] = 0;
  }
  bratio[0] = 50.;
  mode[0][0] = 1000020030; // Helium3
  mode[0][1] = -211;       // negative pion

  bratio[1] = 50.;
  mode[1][0] = 1000010020; // deuteron
  mode[1][1] = 2212;       // proton
  mode[1][2] = -211;       // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1010010030, bratio, mode);

  // Define the 2- and 3-body phase space decay for the Anti-Hyper-Triton
  Int_t amode[6][3];
  Float_t abratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio[kz] = 0.;
    amode[kz][0] = 0;
    amode[kz][1] = 0;
    amode[kz][2] = 0;
  }
  abratio[0] = 50.;
  amode[0][0] = -1000020030; // anti- Helium3
  amode[0][1] = 211;         // positive pion
  abratio[1] = 50.;
  amode[1][0] = -1000010020; // anti-deuteron
  amode[1][1] = -2212;       // anti-proton
  amode[1][2] = 211;         // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1010010030, abratio, amode);

  ////// ----------Hypernuclei with Mass=4 ----------- //////////

  // Define the 2- and 3-body phase space decay for the Hyper Hydrogen 4

  Int_t mode3[6][3];
  Float_t bratio3[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio3[kz] = 0.;
    mode3[kz][0] = 0;
    mode3[kz][1] = 0;
    mode3[kz][2] = 0;
  }
  bratio3[0] = 50.;
  mode3[0][0] = 1000020040; // Helium4
  mode3[0][1] = -211;       // negative pion

  bratio3[1] = 50.;
  mode3[1][0] = 1000010030; // tritium
  mode3[1][1] = 2212;       // proton
  mode3[1][2] = -211;       // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1010010040, bratio3, mode3);
  //Decay for the excited state (after em transition)
  TVirtualMC::GetMC()->SetDecayMode(1010010041, bratio3, mode3);

  // Define the 2- and 3-body phase space decay for the Hyper Hydrogen 4
  Int_t amode3[6][3];
  Float_t abratio3[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio3[kz] = 0.;
    amode3[kz][0] = 0;
    amode3[kz][1] = 0;
    amode3[kz][2] = 0;
  }
  abratio3[0] = 50.;
  amode3[0][0] = -1000020040; // anti- Helium4
  amode3[0][1] = 211;         // positive pion
  abratio3[1] = 50.;
  amode3[1][0] = -1000010030; // anti-tritium
  amode3[1][1] = -2212;       // anti-proton
  amode3[1][2] = 211;         // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1010010040, abratio3, amode3);
  //Decay for the excited state (after em transition)
  TVirtualMC::GetMC()->SetDecayMode(-1010010041, abratio3, amode3);

  // Define the 3-body phase space decay for the Hyper Helium 4
  Int_t mode4[6][3];
  Float_t bratio4[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio4[kz] = 0.;
    mode4[kz][0] = 0;
    mode4[kz][1] = 0;
    mode4[kz][2] = 0;
  }
  bratio4[0] = 100.;
  mode4[0][0] = 1000020030; // Helium3
  mode4[0][1] = -211;       // negative pion
  mode4[0][2] = 2212;       // proton

  TVirtualMC::GetMC()->SetDecayMode(1010020040, bratio4, mode4);
  //Decay for the excited state (after em transition)
  TVirtualMC::GetMC()->SetDecayMode(1010020041, bratio4, mode4);

  // Define the 2-body phase space decay for the Anti-Hyper Helium 4
  Int_t amode4[6][3];
  Float_t abratio4[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio4[kz] = 0.;
    amode4[kz][0] = 0;
    amode4[kz][1] = 0;
    amode4[kz][2] = 0;
  }
  abratio4[0] = 100.;
  amode4[0][0] = -1000020030; // anti-Helium 3
  amode4[0][1] = 211;         // positive pion
  amode4[0][2] = -2212;       // anti proton

  TVirtualMC::GetMC()->SetDecayMode(-1010020040, abratio4, amode4);
  //Decay for the excited state (after em transition)
  TVirtualMC::GetMC()->SetDecayMode(-1010020041, abratio4, amode4);

  // Define the 2-body phase space decay for the Lithium 4
  Int_t model4[6][3];
  Float_t bratiol4[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratiol4[kz] = 0.;
    model4[kz][0] = 0;
    model4[kz][1] = 0;
    model4[kz][2] = 0;
  }
  bratiol4[0] = 100.;
  model4[0][0] = 1000020030; // Helium3
  model4[0][1] = 2212;       // proton

  TVirtualMC::GetMC()->SetDecayMode(1000030040, bratiol4, model4);

  // Define the 2-body phase space decay for the Anti-Lithium 4
  Int_t amodel4[6][3];
  Float_t abratiol4[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratiol4[kz] = 0.;
    amodel4[kz][0] = 0;
    amodel4[kz][1] = 0;
    amodel4[kz][2] = 0;
  }
  abratiol4[0] = 100.;
  amodel4[0][0] = -1000020030; // Anti-Helium3
  amodel4[0][1] = -2212;       // Anti-proton

  TVirtualMC::GetMC()->SetDecayMode(-1000030040, abratiol4, amodel4);

  // Define the 3-body phase space decay for the Hyper Helium 5
  Int_t mode41[6][3];
  Float_t bratio41[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio41[kz] = 0.;
    mode41[kz][0] = 0;
    mode41[kz][1] = 0;
    mode41[kz][2] = 0;
  }
  bratio41[0] = 50.;
  mode41[0][0] = 1000020040; // Helium4
  mode41[0][1] = -211;       // negative pion
  mode41[0][2] = 2212;       // proton
  bratio41[1] = 50.;
  mode41[1][0] = 1000020030; // Helium3
  mode41[1][1] = -211;       // negative pion
  mode41[1][2] = 1000010020; // Deuteron

  TVirtualMC::GetMC()->SetDecayMode(1010020050, bratio41, mode41);

  // Define the 2-body phase space decay for the Anti-Hyper Helium 5
  Int_t amode41[6][3];
  Float_t abratio41[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio41[kz] = 0.;
    amode41[kz][0] = 0;
    amode41[kz][1] = 0;
    amode41[kz][2] = 0;
  }
  abratio41[0] = 50.;
  amode41[0][0] = -1000020040; // anti-Helium 4
  amode41[0][1] = 211;         // positive pion
  amode41[0][2] = -2212;       // anti proton
  abratio41[1] = 50.;
  amode41[1][0] = -1000020030; // anti-Helium 3
  amode41[1][1] = 211;         // positive pion
  amode41[1][2] = -1000010020; // anti deuteron

  TVirtualMC::GetMC()->SetDecayMode(-1010020050, abratio41, amode41);

  // Define the 3-body phase space decay for the Double Hyper Hydrogen 4
  Int_t mode42[6][3];
  Float_t bratio42[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio42[kz] = 0.;
    mode42[kz][0] = 0;
    mode42[kz][1] = 0;
    mode42[kz][2] = 0;
  }
  bratio42[0] = 100.;
  mode42[0][0] = 1010020040; // Hyper-Helium4
  mode42[0][1] = -211;       // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1020010040, bratio42, mode42);

  // Define the 2-body phase space decay for the Anti Double Hyper Hydrogen 4
  Int_t amode42[6][3];
  Float_t abratio42[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio42[kz] = 0.;
    amode42[kz][0] = 0;
    amode42[kz][1] = 0;
    amode42[kz][2] = 0;
  }
  abratio42[0] = 100.;
  amode42[0][0] = -1010020040; // anti-Hyper-Helium 4
  amode42[0][1] = 211;         // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1020010040, abratio42, amode42);

  // Define the 2-body phase space decay for the Lambda-neutron boundstate
  Int_t mode1[6][3];
  Float_t bratio1[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio1[kz] = 0.;
    mode1[kz][0] = 0;
    mode1[kz][1] = 0;
    mode1[kz][2] = 0;
  }
  bratio1[0] = 100.;
  mode1[0][0] = 1000010020; // deuteron
  mode1[0][1] = -211;       // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1010000020, bratio1, mode1);

  // Define the 2-body phase space decay for the Anti-Lambda-neutron boundstate
  Int_t amode1[6][3];
  Float_t abratio1[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio1[kz] = 0.;
    amode1[kz][0] = 0;
    amode1[kz][1] = 0;
    amode1[kz][2] = 0;
  }
  abratio1[0] = 100.;
  amode1[0][0] = -1000010020; // anti-deuteron
  amode1[0][1] = 211;         // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1010000020, abratio1, amode1);

  // Define the 2-body phase space decay for the H-Dibaryon
  Int_t mode2[6][3];
  Float_t bratio2[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio2[kz] = 0.;
    mode2[kz][0] = 0;
    mode2[kz][1] = 0;
    mode2[kz][2] = 0;
  }
  bratio2[0] = 100.;
  mode2[0][0] = 3122; // Lambda
  mode2[0][1] = 2212; // proton
  mode2[0][2] = -211; // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1020000020, bratio2, mode2);

  // Define the 2-body phase space decay for the Anti-H-Dibaryon
  Int_t amode2[6][3];
  Float_t abratio2[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio2[kz] = 0.;
    amode2[kz][0] = 0;
    amode2[kz][1] = 0;
    amode2[kz][2] = 0;
  }
  abratio2[0] = 100.;
  amode2[0][0] = -3122; // anti-deuteron
  amode2[0][1] = -2212; // anti-proton
  amode2[0][2] = 211;   // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1020000020, abratio2, amode2);

  // Define the 2-body phase space decay for the Xi0P
  Int_t mode5[6][3];
  Float_t bratio5[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio5[kz] = 0.;
    mode5[kz][0] = 0;
    mode5[kz][1] = 0;
    mode5[kz][2] = 0;
  }
  bratio5[0] = 100.;
  mode5[0][0] = 3122; // Lambda
  mode5[0][1] = 2212; // proton

  TVirtualMC::GetMC()->SetDecayMode(1020010020, bratio5, mode5);

  // Define the 2-body phase space decay for the Anti-Xi0P
  Int_t amode5[6][3];
  Float_t abratio5[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio5[kz] = 0.;
    amode5[kz][0] = 0;
    amode5[kz][1] = 0;
    amode5[kz][2] = 0;
  }
  abratio5[0] = 100.;
  amode5[0][0] = -3122; // anti-Lambda
  amode5[0][1] = -2212; // anti-proton

  TVirtualMC::GetMC()->SetDecayMode(-1020010020, abratio5, amode5);

  // Define the 2-body phase space decay for the Lambda-Neutron-Neutron
  Int_t mode6[6][3];
  Float_t bratio6[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio6[kz] = 0.;
    mode6[kz][0] = 0;
    mode6[kz][1] = 0;
    mode6[kz][2] = 0;
  }
  bratio6[0] = 100.;
  mode6[0][0] = 1000010030; // triton
  mode6[0][1] = -211;       // pion

  TVirtualMC::GetMC()->SetDecayMode(1010000030, bratio6, mode6);

  // Define the 2-body phase space decay for the Anti-Lambda-Neutron-Neutron
  Int_t amode6[6][3];
  Float_t abratio6[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio6[kz] = 0.;
    amode6[kz][0] = 0;
    amode6[kz][1] = 0;
    amode6[kz][2] = 0;
  }
  abratio6[0] = 100.;
  amode6[0][0] = -1000010030; // anti-triton
  amode6[0][1] = 211;         // pion

  TVirtualMC::GetMC()->SetDecayMode(-1010000030, abratio6, amode6);

  // Define the 3-body phase space decay for the Omega-Proton
  Int_t mode7[6][3];
  Float_t bratio7[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio7[kz] = 0.;
    mode7[kz][0] = 0;
    mode7[kz][1] = 0;
    mode7[kz][2] = 0;
  }
  bratio7[0] = 100.;
  mode7[0][0] = 3122; // Lambda
  mode7[0][1] = -321; // negative Kaon
  mode7[0][2] = 2212; // proton

  TVirtualMC::GetMC()->SetDecayMode(1030000020, bratio7, mode7);

  // Define the 3-body phase space decay for the Anti-Omega-Proton
  Int_t amode7[6][3];
  Float_t abratio7[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio7[kz] = 0.;
    amode7[kz][0] = 0;
    amode7[kz][1] = 0;
    amode7[kz][2] = 0;
  }
  abratio7[0] = 100.;
  amode7[0][0] = -3122; // anti-Lambda
  amode7[0][1] = 321;   // positive kaon
  amode7[0][2] = -2212; // anti-proton

  TVirtualMC::GetMC()->SetDecayMode(-1030000020, abratio7, amode7);

  // Define the 2-body phase space decay for the Omega-Neutron
  Int_t mode8[6][3];
  Float_t bratio8[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio8[kz] = 0.;
    mode8[kz][0] = 0;
    mode8[kz][1] = 0;
    mode8[kz][2] = 0;
  }
  bratio8[0] = 100.;
  mode8[0][0] = 3122; // Lambda
  mode8[0][1] = 3312; // negative Xi

  TVirtualMC::GetMC()->SetDecayMode(1030010020, bratio8, mode8);

  // Define the 2-body phase space decay for the Anti-Omega-Neutron
  Int_t amode8[6][3];
  Float_t abratio8[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio8[kz] = 0.;
    amode8[kz][0] = 0;
    amode8[kz][1] = 0;
    amode8[kz][2] = 0;
  }
  abratio8[0] = 100.;
  amode8[0][0] = -3122; // anti-Lambda
  amode8[0][1] = -3312; // positive Xi

  TVirtualMC::GetMC()->SetDecayMode(-1030010020, abratio8, amode8);

  // Define the 3-body phase space decay for the Omega-Omega
  Int_t mode9[6][3];
  Float_t bratio9[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio9[kz] = 0.;
    mode9[kz][0] = 0;
    mode9[kz][1] = 0;
    mode9[kz][2] = 0;
  }
  bratio9[0] = 100.;
  mode9[0][0] = 3334; // negative Omega
  mode9[0][1] = 3312; // negative Xi

  TVirtualMC::GetMC()->SetDecayMode(1060020020, bratio9, mode9);

  // Define the 3-body phase space decay for the Anti-Omega-Omega
  Int_t amode9[6][3];
  Float_t abratio9[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio9[kz] = 0.;
    amode9[kz][0] = 0;
    amode9[kz][1] = 0;
    amode9[kz][2] = 0;
  }
  abratio9[0] = 100.;
  amode9[0][0] = -3334; // positive Omega
  amode9[0][1] = -3312; // positive Xi

  TVirtualMC::GetMC()->SetDecayMode(-1060020020, abratio9, amode9);

  // Define the 2- and 3-body phase space decay for the Lambda(1405)-Proton
  Int_t mode10[6][3];
  Float_t bratio10[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio10[kz] = 0.;
    mode10[kz][0] = 0;
    mode10[kz][1] = 0;
    mode10[kz][2] = 0;
  }
  bratio10[0] = 50.;
  mode10[0][0] = 3122; // Lambda
  mode10[0][1] = 2212; // proton
  bratio10[1] = 50.;
  mode10[1][0] = 2212; // proton
  mode10[1][1] = -321; // negative kaon
  mode10[1][2] = 2212; // proton

  TVirtualMC::GetMC()->SetDecayMode(1010010021, bratio10, mode10);

  // Define the 2- and 3-body phase space decay for the Anti-Lambda(1405)-Proton
  Int_t amode10[6][3];
  Float_t abratio10[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio10[kz] = 0.;
    amode10[kz][0] = 0;
    amode10[kz][1] = 0;
    amode10[kz][2] = 0;
  }
  abratio10[0] = 50.;
  amode10[0][0] = -3122; // anti-Lambda
  amode10[0][1] = -2212; // anti-proton
  abratio10[1] = 50.;
  amode10[1][0] = -2212; // anti-proton
  amode10[1][1] = 321;   // positive kaon
  amode10[1][2] = -2212; // anti-proton

  TVirtualMC::GetMC()->SetDecayMode(-1010010021, abratio10, amode10);

  // Define the 3-body phase space decay for the Lambda(1405)-Lambda(1405)
  Int_t mode11[6][3];
  Float_t bratio11[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio11[kz] = 0.;
    mode11[kz][0] = 0;
    mode11[kz][1] = 0;
    mode11[kz][2] = 0;
  }
  bratio11[0] = 50.;
  mode11[0][0] = 3122; // Lambda
  mode11[0][1] = 3122; // Lambda
  bratio11[1] = 50.;
  mode11[1][0] = 3122; // Lambda
  mode11[1][1] = 2212; // proton
  mode11[1][2] = -211; // negative pion

  TVirtualMC::GetMC()->SetDecayMode(1020000021, bratio11, mode11);

  // Define the 3-body phase space decay for the Anti-Lambda(1405)-Lambda(1405)
  Int_t amode11[6][3];
  Float_t abratio11[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    abratio11[kz] = 0.;
    amode11[kz][0] = 0;
    amode11[kz][1] = 0;
    amode11[kz][2] = 0;
  }
  abratio11[0] = 50.;
  amode11[0][0] = -3122; // anti-Lambda
  amode11[0][1] = -3122; // anti-Lambda
  abratio11[1] = 50.;
  amode11[1][0] = -3122; // anti-Lambda
  amode11[1][1] = -2212; // anti-proton
  amode11[1][2] = 211;   // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-1020000021, abratio11, amode11);

  // Define the decays for the c-triton
  Int_t ctmode[6][3];
  Float_t ctbratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    ctbratio[kz] = 0.;
    ctmode[kz][0] = 0;
    ctmode[kz][1] = 0;
    ctmode[kz][2] = 0;
  }
  ctbratio[0] = 50.;
  ctmode[0][0] = 1000020030; // Helium3
  ctmode[0][1] = 310;        // K0s

  ctbratio[1] = 50.;
  ctmode[1][0] = 1000020030; // Helium3
  ctmode[1][1] = -321;       // negative kaon
  ctmode[1][2] = 211;        // positive pion

  TVirtualMC::GetMC()->SetDecayMode(2010010030, ctbratio, ctmode);

  // Define the decays for the anti-c-triton
  Int_t actmode[6][3];
  Float_t actbratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    actbratio[kz] = 0.;
    actmode[kz][0] = 0;
    actmode[kz][1] = 0;
    actmode[kz][2] = 0;
  }
  actbratio[0] = 50.;
  actmode[0][0] = -1000020030; // Helium3
  actmode[0][1] = 310;         // K0s

  actbratio[1] = 50.;
  actmode[1][0] = -1000020030; // Helium3
  actmode[1][1] = 321;         // negative kaon
  actmode[1][2] = -211;        // positive pion

  TVirtualMC::GetMC()->SetDecayMode(-2010010030, actbratio, actmode);

  // Define the decays for the c-deuteron
  Int_t cdmode[6][3];
  Float_t cdbratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    cdbratio[kz] = 0.;
    cdmode[kz][0] = 0;
    cdmode[kz][1] = 0;
    cdmode[kz][2] = 0;
  }
  cdbratio[0] = 50.;
  cdmode[0][0] = 1000010020; // deuteron
  cdmode[0][1] = -321;       // negative kaon
  cdmode[0][2] = 211;        // positive pion

  cdbratio[1] = 50.;
  cdmode[1][0] = 1000010020; // deuteron
  cdmode[1][1] = 310;        // K0s

  TVirtualMC::GetMC()->SetDecayMode(2010010020, cdbratio, cdmode);

  // Define the decays for the anti-c-deuteron
  Int_t acdmode[6][3];
  Float_t acdbratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    acdbratio[kz] = 0.;
    acdmode[kz][0] = 0;
    acdmode[kz][1] = 0;
    acdmode[kz][2] = 0;
  }
  acdbratio[0] = 50.;
  acdmode[0][0] = -1000010020; // deuteron
  acdmode[0][1] = 321;         // negative kaon
  acdmode[0][2] = -211;        // positive pion

  acdbratio[1] = 50.;
  acdmode[1][0] = -1000010020; // deuteron
  acdmode[1][1] = 310;         // K0s

  TVirtualMC::GetMC()->SetDecayMode(-2010010020, acdbratio, acdmode);

  ///////////////////////////////////////////////////////////////////

  // Define the 2-body phase space decay for the f0(980)
  //  Int_t mode[6][3];
  //  Float_t bratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio[kz] = 0.;
    mode[kz][0] = 0;
    mode[kz][1] = 0;
    mode[kz][2] = 0;
  }
  bratio[0] = 100.;
  mode[0][0] = 211;  // pion
  mode[0][1] = -211; // pion

  TVirtualMC::GetMC()->SetDecayMode(9010221, bratio, mode);

  // Define the 2-body phase space decay for the f2(1270)
  //  Int_t mode[6][3];
  //  Float_t bratio[6];

  for (Int_t kz = 0; kz < 6; kz++) {
    bratio[kz] = 0.;
    mode[kz][0] = 0;
    mode[kz][1] = 0;
    mode[kz][2] = 0;
  }
  bratio[0] = 100.;
  mode[0][0] = 211;  // pion
  mode[0][1] = -211; // pion

  TVirtualMC::GetMC()->SetDecayMode(225, bratio, mode);

  // Lambda1520/Lambda1520bar

  TVirtualMC::GetMC()->DefineParticle(3124, "Lambda1520", kPTNeutron, 1.5195, 0.0, 4.22e-23, "Hadron", 0.0156, 3, -1, 0, 0, 0, 0, 0, 1, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-3124, "Lambda1520bar", kPTNeutron, 1.5195, 0.0, 4.22e-23, "Hadron", 0.0156, 3, -1, 0, 0, 0, 0, 0, -1, kTRUE);

  // Lambda1520 decay modes

  // L(1520) -> p K-
  bratio[0] = 0.223547;
  mode[0][0] = 2212;
  mode[0][1] = -321;

  // L(1520) -> n K0
  bratio[1] = 0.223547;
  mode[1][0] = 2112;
  mode[1][1] = -311;

  // L(1520) -> Sigma+ pi-
  bratio[2] = 0.139096;
  mode[2][0] = 3222;
  mode[2][1] = -211;

  // L(1520) -> Sigma0 pi0
  bratio[3] = 0.139096;
  mode[3][0] = 3212;
  mode[3][1] = 111;

  // L(1520) -> Sigma- pi+
  bratio[4] = 0.139096;
  mode[4][0] = 3112;
  mode[4][1] = 211;

  // The other decay modes are neglected
  bratio[5] = 0.;
  mode[5][0] = 0;
  mode[5][1] = 0;

  TVirtualMC::GetMC()->SetDecayMode(3124, bratio, mode);

  // Lambda1520bar decay modes

  // L(1520)bar -> p- K+
  bratio[0] = 0.223547;
  mode[0][0] = -2212;
  mode[0][1] = 321;

  // L(1520)bar -> nbar K0bar
  bratio[1] = 0.223547;
  mode[1][0] = -2112;
  mode[1][1] = 311;

  // L(1520)bar -> Sigmabar- pi+
  bratio[2] = 0.139096;
  mode[2][0] = -3222;
  mode[2][1] = 211;

  // L(1520)bar -> Sigma0bar pi0
  bratio[3] = 0.139096;
  mode[3][0] = -3212;
  mode[3][1] = 111;

  // L(1520)bar -> Sigmabar+ pi-
  bratio[4] = 0.139096;
  mode[4][0] = -3112;
  mode[4][1] = -211;

  // The other decay modes are neglected
  bratio[5] = 0.;
  mode[5][0] = 0;
  mode[5][1] = 0;

  TVirtualMC::GetMC()->SetDecayMode(-3124, bratio, mode);

  // --------------------------------------------------------------------

  //Sexaquark (uuddss): compact, neutral and stable hypothetical bound state (arxiv.org/abs/1708.08951)
  TVirtualMC::GetMC()->DefineParticle(900000020, "Sexaquark", kPTUndefined, 2.0, 0.0, 4.35e+17, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, 2, kTRUE);
  TVirtualMC::GetMC()->DefineParticle(-900000020, "AntiSexaquark", kPTUndefined, 2.0, 0.0, 4.35e+17, "Hadron", 0.0, 0, 1, 0, 0, 0, 0, 0, -2, kTRUE);
}

void O2MCApplicationBase::AddParticles()
{
  // dispatch first to function in FairRoot
  FairMCApplication::AddParticles();

  // register special particles for ALICE
  // TODO: try to make use of FairRoot if easier or more customizable
  addSpecialParticles();

  auto& param = o2::conf::SimUserDecay::Instance();
  LOG(info) << "Printing \'SimUserDecay\' parameters";
  LOG(info) << param;

  // check if there are PDG codes requested for user decay
  if (param.pdglist.empty()) {
    return;
  }

  // loop over PDG codes in the string
  std::stringstream ss(param.pdglist);
  int pdg;
  while (ss >> pdg) {
    LOG(info) << "Setting user decay for PDG " << pdg;
    TVirtualMC::GetMC()->SetUserDecay(pdg);
  }
}

void O2MCApplicationBase::initTrackRefHook()
{
  if (mCutParams.stepTrackRefHook) {
    LOG(info) << "Initializing the hook for TrackReferences during stepping";
    auto expandedTrackRefHookFileName = o2::utils::expandShellVarsInFileName(mCutParams.stepTrackRefHookFile);
    if (std::filesystem::exists(expandedTrackRefHookFileName)) {
      // if this file exists we will compile the hook on the fly
      mTrackRefFcn = o2::conf::GetFromMacro<TrackRefFcn>(mCutParams.stepTrackRefHookFile, "trackRefHook()", "o2::steer::O2MCApplicationBase::TrackRefFcn", "o2mc_stepping_trackref_hook");
      LOG(info) << "Hook initialized from file " << expandedTrackRefHookFileName;
    } else {
      LOG(error) << "Did not file TrackRefHook file " << expandedTrackRefHookFileName << " ; Will not execute hook";
      mTrackRefFcn = [](TVirtualMC const*) {}; // do nothing
    }
  }
}

void O2MCApplication::initLate()
{
  o2::utils::ShmManager::Instance().occupySegment();
  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->initializeLate();
    }
  }
}

void O2MCApplication::attachSubEventInfo(fair::mq::Parts& parts, o2::data::SubEventInfo const& info) const
{
  // parts.AddPart(std::move(mSimDataChannel->NewSimpleMessage(info)));
  o2::base::attachTMessage(info, *mSimDataChannel, parts);
}

// helper function to fetch data from FairRootManager branch and serialize it
// returns handle to container
template <typename T>
const T* attachBranch(std::string const& name, fair::mq::Channel& channel, fair::mq::Parts& parts)
{
  auto mgr = FairRootManager::Instance();
  // check if branch is present
  if (mgr->GetBranchId(name) == -1) {
    LOG(error) << "Branch " << name << " not found";
    return nullptr;
  }
  auto data = mgr->InitObjectAs<const T*>(name.c_str());
  if (data) {
    o2::base::attachTMessage(*data, channel, parts);
  }
  return data;
}

void O2MCApplication::setSubEventInfo(o2::data::SubEventInfo* i)
{
  mSubEventInfo = i;
  // being communicated a SubEventInfo also means we get a FairMCEventHeader
  fMCEventHeader = &mSubEventInfo->mMCEventHeader;
}

void O2MCApplication::SendData()
{
  fair::mq::Parts simdataparts;

  // fill these parts ... the receiver has to unpack similary
  // TODO: actually we could just loop over branches in FairRootManager at this moment?
  mSubEventInfo->npersistenttracks = static_cast<o2::data::Stack*>(GetStack())->getMCTracks()->size();
  mSubEventInfo->nprimarytracks = static_cast<o2::data::Stack*>(GetStack())->GetNprimary();
  attachSubEventInfo(simdataparts, *mSubEventInfo);
  auto tracks = attachBranch<std::vector<o2::MCTrack>>("MCTrack", *mSimDataChannel, simdataparts);
  attachBranch<std::vector<o2::TrackReference>>("TrackRefs", *mSimDataChannel, simdataparts);
  assert(tracks->size() == mSubEventInfo->npersistenttracks);

  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->attachHits(*mSimDataChannel, simdataparts);
    }
  }
  LOG(info) << "sending message with " << simdataparts.Size() << " parts";
  mSimDataChannel->Send(simdataparts);
}
} // namespace steer
} // namespace o2
