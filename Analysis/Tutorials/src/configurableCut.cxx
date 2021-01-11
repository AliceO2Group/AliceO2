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

std::ostream& operator<<(std::ostream& os, configurableCut const& c)
{
  os << "Cut value: " << c.getCut() << "; state: " << c.getState();
  return os;
}

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

void configurableCut::setBins(std::vector<float> bins_)
{
  bins = std::move(bins_);
}

std::vector<float> configurableCut::getBins() const
{
  return bins;
}

void configurableCut::setLabels(std::vector<std::string> labels_)
{
  labels = std::move(labels_);
}

std::vector<std::string> configurableCut::getLabels() const
{
  return labels;
}

void configurableCut::setCuts(o2::framework::Array2D<double> cuts_)
{
  cuts = std::move(cuts_);
}

o2::framework::Array2D<double> configurableCut::getCuts() const
{
  return cuts;
}
