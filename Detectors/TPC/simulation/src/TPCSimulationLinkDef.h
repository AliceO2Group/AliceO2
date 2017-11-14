// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::TPC::Baseline+;
#pragma link C++ class o2::TPC::CommonModeContainer+;
#pragma link C++ class o2::TPC::Detector+;
#pragma link C++ class o2::Base::DetImpl<o2::TPC::Detector>+;
#pragma link C++ class o2::TPC::DigitMCMetaData+;
#pragma link C++ class std::vector<o2::TPC::DigitMCMetaData>+;
#pragma link C++ class o2::TPC::DigitContainer+;
#pragma link C++ class o2::TPC::DigitCRU+;
#pragma link C++ class o2::TPC::Digitizer+;
#pragma link C++ class o2::TPC::DigitizerTask+;
#pragma link C++ class o2::TPC::DigitPad+;
#pragma link C++ class o2::TPC::DigitRow+;
#pragma link C++ class o2::TPC::DigitTime+;
#pragma link C++ class o2::TPC::ElectronTransport+;
#pragma link C++ class o2::TPC::GEMAmplification+;
#pragma link C++ class o2::TPC::PadResponse+;
#pragma link C++ class o2::TPC::Point+;
#pragma link C++ class o2::TPC::ElementalHit+;
#pragma link C++ class std::vector<o2::TPC::ElementalHit>+;
#pragma link C++ class o2::TPC::HitGroup+;
#pragma link C++ class o2::TPC::SAMPAProcessing+;

#pragma link C++ class std::vector<o2::TPC::Cluster>+;
#pragma link C++ class std::vector<o2::TPC::BoxCluster>+;
#pragma link C++ class std::vector<o2::TPC::HwCluster>+;
#pragma link C++ class std::vector<o2::TPC::HitGroup>+;

#endif
