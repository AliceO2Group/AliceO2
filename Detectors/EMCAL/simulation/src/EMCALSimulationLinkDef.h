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

#pragma link C++ class o2::emcal::Detector + ;
#pragma link C++ class o2::base::DetImpl < o2::emcal::Detector> + ;
#pragma link C++ class o2::emcal::Digitizer + ;
#pragma link C++ class o2::emcal::SDigitizer + ;
#pragma link C++ class o2::emcal::DigitsWriteoutBuffer + ;
#pragma link C++ class o2::emcal::SimParam + ;
#pragma link C++ class o2::emcal::LabeledDigit + ;
#pragma link C++ class o2::emcal::RawWriter + ;

#pragma link C++ class std::list < o2::emcal::LabeledDigit> + ;

#endif
