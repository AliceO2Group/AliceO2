// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iomanip>

#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TList.h>

#include <FairLogger.h>

#include "PHOSBase/Geometry.h"
//#include "PHOSBase/ShishKebabTrd1Module.h"

//#include <boost/algorithm/string/predicate.hpp>

using namespace o2::PHOS;

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;
