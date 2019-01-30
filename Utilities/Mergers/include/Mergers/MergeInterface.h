// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGEINTERFACE_H
#define ALICEO2_MERGEINTERFACE_H

/// \file MergeInterface.h
/// \brief Definition of O2 Mergers merging interface, v1.0
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <vector>
#include <TObject.h>
#include <TCollection.h>

namespace o2
{
namespace mergers
{

/// \brief Interface allowing custom behaviour of Mergers components.
///
/// Interface allowing custom behaviour of Mergers components - unpacking the object into
/// vector of objects, merging objects and providing timestamp of the object. Keep in mind,
/// that you aside from implementing needed functions, one should activate them in MergerConfig.
class MergeInterface
{
 public:
  virtual ~MergeInterface();

  /// \brief Custom unpacking function.
  virtual std::vector<TObject*> unpack()
  {
    return {};
  };

  /// \brief Custom merge function.
  virtual Long64_t merge(TCollection* list)
  {
    return 0;
  };

  /// \brief Timestamp getter function.
  virtual double time() // getTime?
  {
    return 0;
  };
  ClassDef(MergeInterface, 1);
};

} // namespace mergers
} // namespace o2

#endif //ALICEO2_MERGEINTERFACE_H
