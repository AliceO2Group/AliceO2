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

#pragma link C++ class o2::tpc::CalibRawBase;
#pragma link C++ class o2::tpc::CalibPedestal;
#pragma link C++ class o2::tpc::CalibPedestalParam +;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPedestalParam> + ;
#pragma link C++ class o2::tpc::CalibPulser;
#pragma link C++ class o2::tpc::CalibPulserParam +;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPulserParam> + ;
#pragma link C++ class o2::tpc::CalibTreeDump;
#pragma link C++ class o2::tpc::DigitDump;
#pragma link C++ class o2::tpc::CalibPadGainTracks;
#pragma link C++ class o2::tpc::FastHisto<float> +;
#pragma link C++ class o2::tpc::FastHisto<unsigned int> +;

#endif
