// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ANALYSISDATAMODELHELPERS_H_
#define O2_FRAMEWORK_ANALYSISDATAMODELHELPERS_H_

#include "Framework/AnalysisDataModel.h"
#include "Headers/DataHeader.h"

namespace o2::aod::datamodel
{
std::string getTreeName(header::DataHeader dh);
std::vector<std::string> getColumnNames(header::DataHeader dh);
} // namespace o2::aod::datamodel
#endif // O2_FRAMEWORK_ANALYSISDATAMODELHELPERS_H_
