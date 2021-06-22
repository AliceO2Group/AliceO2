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

#pragma link C++ class o2::phos::PHOSPedestalCalibDevice + ;
#pragma link C++ class o2::phos::PHOSHGLGRatioCalibDevice + ;
#pragma link C++ class o2::phos::ETCalibHistos + ;
#pragma link C++ class o2::phos::PHOSEnergySlot + ;
#pragma link C++ class o2::phos::PHOSEnergyCalibrator + ;
#pragma link C++ class o2::phos::PHOSEnergyCalibDevice + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::phos::PHOSEnergySlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::phos::Cluster, o2::phos::PHOSEnergySlot> + ;
#pragma link C++ class o2::phos::TurnOnHistos + ;
#pragma link C++ class o2::phos::PHOSTurnonSlot + ;
#pragma link C++ class o2::phos::PHOSTurnonCalibrator + ;
#pragma link C++ class o2::phos::PHOSTurnonCalibDevice + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::phos::PHOSTurnonSlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::phos::Cluster, o2::phos::PHOSTurnonSlot> + ;
#pragma link C++ class o2::phos::PHOSRunbyrunSlot + ;
#pragma link C++ class o2::phos::PHOSRunbyrunCalibrator + ;
#pragma link C++ class o2::phos::PHOSRunbyrunCalibDevice + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::phos::PHOSRunbyrunSlot> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::phos::Cluster, o2::phos::PHOSRunbyrunSlot> + ;

#endif
