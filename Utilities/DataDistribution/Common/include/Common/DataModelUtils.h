// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATAMODEL_UTILS_H_
#define ALICEO2_DATAMODEL_UTILS_H_

namespace o2 {
namespace DataDistribution {

////////////////////////////////////////////////////////////////////////////////
/// IDataModelObject interface
////////////////////////////////////////////////////////////////////////////////
class O2SubTimeFrameLinkData;
class O2SubTimeFrameCruData;
class O2SubTimeFrameRawData;

class SubTimeFrameDataSource;
class O2SubTimeFrame;

class ISubTimeFrameVisitor {
public:
  virtual void visit(O2SubTimeFrame&) = 0;
  virtual void visit(SubTimeFrameDataSource&) = 0;
};

class IDataModelObject {
public:
  virtual void accept(ISubTimeFrameVisitor& v) = 0;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_DATAMODEL_UTILS_H_ */
