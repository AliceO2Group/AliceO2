/// \author R+Preghenella - July 2020

/// This Pythia8 UserHooks can veto the processing at parton level.
/// The partonic event is scanned searching for a c-cbar mother
/// with at least one of the c quarks produced withing a fiducial
/// window around midrapidity that can be specified by the user.

#include "Pythia8/Pythia.h"

class UserHooks_ccbar : public Pythia8::UserHooks
{
  
 public:
  UserHooks_ccbar() = default;
  ~UserHooks_ccbar() = default;
  bool canVetoPartonLevel() override { return true; };
  bool doVetoPartonLevel(const Pythia8::Event& event) override {
    // search for c-cbar mother with at least one c at midrapidity
    for (int ipa = 0; ipa < event.size(); ++ipa) {
      auto daughterList = event[ipa].daughterList();
      bool hasc = false, hascbar = false, atmidy = false;
      for (auto ida : daughterList) {
	if (event[ida].id() == 4) hasc = true;
	if (event[ida].id() == -4) hascbar = true;
	if (fabs(event[ida].y()) < mRapidity) atmidy = true;
      }
      if (hasc && hascbar && atmidy)
	return false; // found it, do not veto event
    }
    return true; // did not find it, veto event
  };

  void setRapidity(double val) { mRapidity = val; };
  
private:

  double mRapidity = 1.5;
  
};

Pythia8::UserHooks*
  pythia8_userhooks_ccbar(double rapidity = 1.5)
{
  auto hooks = new UserHooks_ccbar();
  hooks->setRapidity(rapidity);
  return hooks;
}
