// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGEINTERFACE_H
#define ALICEO2_MERGEINTERFACE_H

/// \file MergeInterface.h
/// \brief Definition of O2 Mergers merging interface, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <Rtypes.h>

namespace o2::mergers
{

/// \brief An interface which allows to merge custom objects.
///
/// An interface which allows to merge custom objects.
/// The custom class can inherit from TObject, but this is not an obligation.
class MergeInterface
{
 public:
  // Please make sure to properly delete an object. If the inheriting class object is a container,
  // make sure that all entries are correctly deleted as well.
  virtual ~MergeInterface() = default;

  /// \brief Custom merge method.
  virtual void merge(MergeInterface* const other) = 0; // const argument

  /// \brief Lets the child perform any routines after the object was deserialized (e.g. setting the correct ownership)
  virtual void postDeserialization(){};

  /// \brief Should return an object subset which is supposed to take part in generating moving windows.
  virtual MergeInterface* cloneMovingWindow() const { return nullptr; }

  ClassDef(MergeInterface, 1);
};

} // namespace o2::mergers

#endif //ALICEO2_MERGEINTERFACE_H
