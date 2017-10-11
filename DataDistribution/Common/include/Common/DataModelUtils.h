// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STF_DATAMODEL_UTILS_H_
#define STF_DATAMODEL_UTILS_H_

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// IDataModelObject interface
////////////////////////////////////////////////////////////////////////////////
class EquipmentHBFrames;
class SubTimeFrame;

class ISubTimeFrameVisitor
{
 public:
  virtual void visit(SubTimeFrame&) = 0;
};

class ISubTimeFrameConstVisitor
{
 public:
  virtual void visit(const SubTimeFrame&) = 0;
};

class IDataModelObject
{
 public:
  virtual void accept(ISubTimeFrameVisitor& v) = 0;
  virtual void accept(ISubTimeFrameConstVisitor& v) const = 0;
};
}
} /* o2::DataDistribution */

#endif /* STF_DATAMODEL_UTILS_H_ */
