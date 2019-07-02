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

#pragma link C++ class o2::trd::TRDPadPlane + ;
#pragma link C++ class o2::trd::TRDGeometry + ;
#pragma link C++ class o2::trd::TRDGeometryBase + ;
#pragma link C++ class o2::trd::TRDCommonParam + ;
#pragma link C++ class o2::trd::TRDSimParam + ;
#pragma link C++ class o2::trd::Digit + ;
#pragma link C++ class std::vector<o2::trd::Digit> + ;
#pragma link C++ class o2::trd::TRDFeeParam + ;
#pragma link C++ class o2::trd::TRDCalSingleChamberStatus + ;
#pragma link C++ class o2::trd::TRDCalPadStatus + ;
#pragma link C++ class o2::trd::CalDet + ;
#pragma link C++ class o2::trd::CalROC + ;
#pragma link C++ class o2::trd::PadResponse + ;
#pragma link C++ class o2::trd::MCLabel + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::trd::MCLabel> + ;
#pragma link C++ class o2::trd::LTUParam + ;

#endif
