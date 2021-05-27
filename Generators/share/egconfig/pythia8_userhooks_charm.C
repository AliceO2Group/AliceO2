// Pythia8 UserHooks
//
//   usage: o2sim -g pythia8pp --configKeyValues "GeneratorPythia8.hooksFileName=pythia8_userhooks_charm.C"
//
/// \author R+Preghenella - February 2020

#if defined(__CLING__)
// clang-format off
R__LOAD_LIBRARY(libpythia8)
R__ADD_INCLUDE_PATH($PYTHIA8/include)
// clang-format on
#endif

#include "Pythia8/Pythia.h"

/** This is an example of Pythia8 UserHooks.
 ** The process is queried at the parton level
 ** and it is inhibited unless there is a 
 ** charm (anti)quark produced at |y| < 1.5. **/

class UserHooksCharm : public Pythia8::UserHooks
{
 public:
  UserHooksCharm() = default;
  ~UserHooksCharm() = default;
  bool canVetoPartonLevel() override { return true; };
  bool doVetoPartonLevel(const Pythia8::Event& event) override
  {
    for (int ipa = 0; ipa < event.size(); ++ipa) {
      if (abs(event[ipa].id()) != 4)
        continue;
      if (fabs(event[ipa].y()) > 1.5)
        continue;
      return false;
    }
    return true;
  };
};

Pythia8::UserHooks*
  pythia8_userhooks_charm()
{
  return new UserHooksCharm();
}
