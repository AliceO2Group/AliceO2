// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   Constants.h
/// @author David Rohr
///

#ifndef AliceO2_TPC_Constants_H
#define AliceO2_TPC_Constants_H

namespace o2 { namespace TPC {

class Constants
{
  public:
    // the number of sectors
    static constexpr int MAXSECTOR=36;

    // the number of global pad rows
    static constexpr int MAXGLOBALPADROW=152;
};

}}

#endif
