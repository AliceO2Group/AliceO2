// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/configurableCut.h"
#include <iostream>

bool configurableCut::method(float arg) const
{
  return arg > cut;
}

void configurableCut::setCut(float cut_)
{
  cut = cut_;
}

float configurableCut::getCut() const
{
  return cut;
}

std::ostream& operator<<(std::ostream& os, configurableCut const& c)
{
  os << "Cut value: " << c.getCut() << "; state: " << c.getState();
  return os;
}

void configurableCut::setState(int state_)
{
  state = state_;
}

int configurableCut::getState() const
{
  return state;
}

void configurableCut::setOption(bool option_)
{
  option = option_;
}

bool configurableCut::getOption() const
{
  return option;
}
