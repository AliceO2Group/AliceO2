// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Ex2/A.h"
#include <iostream>

ClassImp(ex2::A);

namespace ex2
{

A::A()
{
  std::cout << "Hello from ex2::A ctor\n";
}
} // namespace ex2
