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

#pragma link C++ class o2::tpc::Detector + ;
#pragma link C++ class o2::base::DetImpl < o2::tpc::Detector> + ;
#pragma link C++ class o2::tpc::DigitMCMetaData + ;
#pragma link C++ class std::vector < o2::tpc::DigitMCMetaData> + ;
#pragma link C++ class o2::tpc::CommonMode + ;
#pragma link C++ class std::vector < o2::tpc::CommonMode> + ;
#pragma link C++ class o2::tpc::DigitContainer + ;
#pragma link C++ class o2::tpc::DigitGlobalPad + ;
#pragma link C++ class o2::tpc::Digitizer + ;
#pragma link C++ class o2::tpc::DigitTime + ;
#pragma link C++ class o2::tpc::ElectronTransport + ;
#pragma link C++ class o2::tpc::GEMAmplification + ;
#pragma link C++ class o2::tpc::PadResponse + ;
#pragma link C++ class o2::tpc::Point + ;
#pragma link C++ class o2::tpc::ElementalHit + ;
#pragma link C++ class std::vector < o2::tpc::ElementalHit> + ;
#pragma link C++ class o2::tpc::HitGroup + ;
#pragma link C++ class o2::tpc::SAMPAProcessing + ;
#pragma link C++ class o2::tpc::SpaceCharge + ;
#pragma link C++ enum o2::tpc::SpaceCharge::SCDistortionType + ;

#pragma link C++ class std::vector < o2::tpc::HitGroup> + ;

#endif
