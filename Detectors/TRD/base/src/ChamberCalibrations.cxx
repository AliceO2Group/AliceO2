// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  ChamberCalibrations class :                                                     //
//    This essentially replicates:  4 caldet, 1 calpad, and                  //
//    Most of the functionality inherint in those classes is present         //
//    with some additional functionality to clean up the interface.          //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TRobustEstimator.h>
#include <sstream>

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/ChamberCalibrations.h"
#include "fairlogger/Logger.h"
using namespace o2::trd;

