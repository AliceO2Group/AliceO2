// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ARROWDEBUGHELPERS_H_
#define O2_FRAMEWORK_ARROWDEBUGHELPERS_H_

// Explicit template declarations to help LLDB debug arrow related parts.

#ifndef NDEBUG
template class std::shared_ptr<arrow::Array>;
template class std::shared_ptr<arrow::ChunkedArray>;
template class std::shared_ptr<arrow::Column>;
template class std::shared_ptr<arrow::Field>;
template class std::shared_ptr<arrow::Schema>;
template class std::shared_ptr<arrow::Table>;
template class std::vector<std::shared_ptr<arrow::Column>>;
template class std::vector<std::shared_ptr<arrow::Field>>;
#endif

#endif // O2_FRAMEWORK_ARROWDEBUGHELPERS_H_
