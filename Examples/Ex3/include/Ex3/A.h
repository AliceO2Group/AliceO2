// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _EX3_A_H_
#define _EX3_A_H_

#include "TObject.h"

namespace ex3
{
class A : public TObject
{
 public:
  A();

  ClassDef(A, 1);
};

} // namespace ex3
#endif
