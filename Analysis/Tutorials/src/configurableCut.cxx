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
