// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_UTILS_H_
#define ALICEO2_SUBTIMEFRAME_UTILS_H_

#include "Common/SubTimeFrameDataModel.h"

#include <Headers/DataHeader.h>

class O2Device;

namespace o2 {
namespace DataDistribution {

////////////////////////////////////////////////////////////////////////////////
/// DataOriginSplitter
////////////////////////////////////////////////////////////////////////////////

class DataIdentifierSplitter : public ISubTimeFrameVisitor {
public:
  DataIdentifierSplitter() = default;
  O2SubTimeFrame split(O2SubTimeFrame& pStf, const DataIdentifier& pDataIdent, const int pChanId);

private:
  void visit(SubTimeFrameDataSource& pStf) override;
  void visit(O2SubTimeFrame& pStf) override;

  DataIdentifier mDataIdentifier;
  O2SubTimeFrame mSubTimeFrame;
};



}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_UTILS_H_ */
