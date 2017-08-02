// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testDetBaseLinkDef.h
/// \brief Linking definitions of Vector class test
/// \author Ruben Shahoyan <ruben.shahoyan@cern.ch>

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::Base::Vector<o2::Base::Track::TrackPar,int>-;
#pragma link C++ class vector<o2::Base::Track::TrackPar>+;

#endif
