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

#pragma link C++ class o2::tof::Geo + ;
#pragma link C++ class o2::tof::Digit + ;
#pragma link C++ class vector < o2::tof::Digit> + ;
#pragma link C++ class o2::tof::Strip + ;
#pragma link C++ class o2::tof::WindowFiller + ;
#pragma link C++ class o2::tof::ReadoutWindowData + ;
#pragma link C++ class vector < o2::tof::ReadoutWindowData> + ;
#pragma link C++ class o2::tof::DigitHeader + ;
#pragma link C++ class vector < o2::tof::DigitHeader> + ;
#pragma link C++ class vector < unsigned int> + ;
#endif
