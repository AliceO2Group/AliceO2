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

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_
#define ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_

#include "Generators/Generator.h"
#include "Pythia8/Pythia.h"
#include <functional>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/
/** Interface to the Pythia8 event generator.  This generator is
 * configured by configuration files (e.g.,
 * `Generators/share/egconfig/pythia8_inel.cfg` for the type of
 * events to generate.
 *
 * The above file, for example, contains
 *
 * @verbatim
 * ### Beams, proton-proton collisions at sqrt(s)=14TeV
 * Beams:idA 2212                  # proton
 * Beams:idB 2212                  # proton
 * Beams:eCM 14000.                # GeV
 *
 * ### Processes, min-bias inelastic events
 * SoftQCD:inelastic on            # all inelastic processes
 *
 * ### Decays. Only decay until 1cm/c - corresponding to (physical) primaries
 * ParticleDecays:limitTau0 on
 * ParticleDecays:tau0Max 10.
 * @endverbatim
 *
 * The configuration file to read is initially set in
 * `GeneratorFactory`, but an additional configuration file can be
 * specified with the configuration key `GeneratorPythia8::config`.
 *
 * If the configuration key `GeneratorPythia8::includePartonEvent` is
 * set to false (default), then the event is pruned.  That is, all
 * particles that are not
 *
 * - beam particles (HepMC status = 4),
 * - decayed particles (HepMC status = 2), nor
 * - final state partcles (HepMC status = 1)
 *
 * are removed from the event before passing on to the simulation.
 * The event structure is kept, so that we have a well-formed event
 * structure.  This reduces the event size by roughly 30%.
 *
 * If the configuration key `GeneratorPythia8::includePartonEvent` is
 * true, then the full event is kept, including intermediate partons
 * such as gluons, pomerons, diffractive hadrons, and so on.
 *
 * In the future, the way to prune events may become more flexible.
 * For example, we could have the configuration keys
 *
 * - GeneratorPythia8::onlyStatus a list of HepMC status codes to accept
 * - GeneratorPythia8::onlyPDGs a list of PDG particle codes to accept
 *
 * The configuration key `GeneratorPythia8::hooksFileName` allows the
 * definition of a Pythia8 user hook.  See for example
 * `Generators/share/egconfig/pythia8_userhooks_charm.C`.  The file
 * specified is interpreted via ROOT (i.e., a ROOT script), and the
 * function name set via the configuration key
 * `GeneratorPythia8::hooksFuncName` (default `pythia8_user_hooks`) is
 * executed.  That function must return a pointer to a
 * `Pythia8::UserHooks` object (see the Pythia8 manual for more on
 * this).
 */
class GeneratorPythia8 : public Generator
{

 public:
  /** default constructor **/
  GeneratorPythia8();
  /** constructor **/
  GeneratorPythia8(const Char_t* name, const Char_t* title = "ALICEo2 Pythia8 Generator");
  /** destructor **/
  ~GeneratorPythia8() override = default;

  /** @{
      @name methods to override **/
  /** Initialize the generator if needed **/
  Bool_t Init() override;
  /** Generate a single event */
  Bool_t generateEvent() override;
  /** Import particles from Pythia onto the simulation event stack */
  Bool_t importParticles() override { return importParticles(mPythia.event); };
  /** @} */

  /** @{
   * @name setters **/
  /** Set Pythia8 configuration file to read */
  void setConfig(std::string val) { mConfig = val; };
  /**
   * Set the ROOT script file name that defines a Pythia8::UserHooks
   * object */
  void setHooksFileName(std::string val) { mHooksFileName = val; };
  /** Function in ROOT script that returns a pointer to a
   * Pythia8::UserHooks object */
  void setHooksFuncName(std::string val) { mHooksFuncName = val; };
  /** Set the user hooks (defined in a Pythia8::UserHooks object) for
   * the event generator. */
  void setUserHooks(Pythia8::UserHooks* hooks);
  /** @} */

  /** @{
   * @name Configuration methods **/
  /** Read a Pythia8 configuration string */
  bool readString(std::string val) { return mPythia.readString(val, true); };
  /** Read a Pythia8 configuration file */
  bool readFile(std::string val) { return mPythia.readFile(val, true); };
  /** @} */

  /** @{
   * @name Utilities **/
  /** Get number of binary collisions.  Note that this method deviates
   * from how the Pythia authors count number of binary collisions */
  void getNcoll(int& nColl)
  {
    getNcoll(mPythia.info, nColl);
  };
  /** Get number of participants.  Note that this method deviates
   * from how the Pythia authors count number of participants */
  void getNpart(int& nPart)
  {
    getNpart(mPythia.info, nPart);
  };
  /** Get number of participants, split by nucleon type and origin.
   * Note that this method deviates from how the Pythia authors count
   * number of participants */
  void getNpart(int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
  {
    getNpart(mPythia.info, nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg);
  };
  /** Get number of nuclei remnants, split by nucleon type and origin. */
  void getNremn(int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg)
  {
    getNremn(mPythia.event, nProtonProj, nNeutronProj, nProtonTarg, nNeutronTarg);
  };
  /** Get number of spectators, split by nucleon type and origin.
   * Note that this method deviates from how the Pythia authors count
   * number of spectators */
  void getNfreeSpec(int& nFreenProj, int& nFreepProj, int& nFreenTarg, int& nFreepTarg)
  {
    getNfreeSpec(mPythia.info, nFreenProj, nFreepProj, nFreenTarg, nFreepTarg);
  };

  typedef std::function<bool(const Pythia8::Particle&)> UserFilterFcn;

 protected:
  /** copy constructor **/
  GeneratorPythia8(const GeneratorPythia8&);
  /** operator= **/
  GeneratorPythia8& operator=(const GeneratorPythia8&);

  /** @{
   * @name Some function definitions
   */
  /** Select particles when pruning event */
  typedef UserFilterFcn Select;
  /** Get relatives (mothers or daughters) of a particle */
  typedef std::vector<int> (*GetRelatives)(const Pythia8::Particle&);
  /** Set relatives (mothers or daughters) of a particle */
  typedef void (*SetRelatives)(Pythia8::Particle&, int, int);
  /** Get range of relatives (mothers or daughters) of a particle */
  typedef std::pair<int, int> (*FirstLastRelative)(const Pythia8::Particle&);
  /** @} */

  /** @{
   * @name Methods that can be overridded **/
  /** Update the event header.  This propagates all sorts of
   * information from Pythia8 to the simulation event header,
   * including parton distribution function parameters, event weight,
   * cross-section information, heavy-ion collision information, and
   * so on. */
  void updateHeader(o2::dataformats::MCEventHeader* eventHeader) override;
  /** @} */

  /** @{
   * @name Internal methods **/
  /** Import particles from Pythia onto the simulation stack */
  Bool_t importParticles(Pythia8::Event& event);
  /** Prune an event.  Only particles for which the function select
   * returns true are kept in the event record.  The structure of the
   * event is preserved, meaning that particles will point back to
   * their ultimate (selected) mothers and, if select preserves the
   * beam particles, the ultimate collision interaction. */
  void pruneEvent(Pythia8::Event& event, Select select);
  /** Investigate relatives (mothers or daughters) for particles to
   * keep when pruning an event.  This checks the current particle,
   * identified by index, if any of its relatives (either mothers or
   * daughters) are to be kept.  If a relative is to be kept, then
   * that is added to an internal list.  If a relative is _not_ to be
   * kept, then that relatives relatives are queried (recursive call).
   * The result of the recursive call is a list of relatives to the
   * current particle which are to be kept.  These are then also added
   * to the internal list.  The relatives that are found to be kept
   * are then set to be relatives of the current particle.  Note that
   * this member function modifies the relatives of the passed
   * particle, and thus modifies the passed event structure.
   * Calculations are cached. */
  void investigateRelatives(Pythia8::Event& event,           // Event
                            const std::vector<int>& old2New, // Map from old to new idx
                            size_t index,                    // Current particle
                            std::vector<bool>& done,         // cache flag
                            GetRelatives getter,             // get relatives
                            SetRelatives setter,             // set relatives
                            FirstLastRelative firstLast,     // get first and last relative
                            const std::string& what,         // what are we looking for
                            const std::string& ind = "");    // logging indent
  /** @{
   * @name utilities **/
  /** Select from ancestor. Fills the output event with all particles
   * related to an ancestor of the input event
   *
   * Starting from ancestor, select all daughters (and their daughters
   * recursively), and store them in the output event.
   **/
  void selectFromAncestor(int ancestor, Pythia8::Event& inputEvent, Pythia8::Event& outputEvent);
  /** Get number of binary collisions.  Note that this method deviates
   * from how the Pythia authors count number of binary collisions */
  void getNcoll(const Pythia8::Info& info, int& nColl);
  /** Get number of participants.  Note that this method deviates
   * from how the Pythia authors count number of participants */
  void getNpart(const Pythia8::Info& info, int& nPart);
  /** Get number of participants, split by nucleon type and origin.
   * Note that this method deviates from how the Pythia authors count
   * number of participants */
  void getNpart(const Pythia8::Info& info, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg);
  /** Get number of nuclei remnants, split by nucleon type and origin. */
  void getNremn(const Pythia8::Event& event, int& nProtonProj, int& nNeutronProj, int& nProtonTarg, int& nNeutronTarg);
  /** Get number of spectators, split by nucleon type and origin.
   * Note that this method deviates from how the Pythia authors count
   * number of spectators */
  void getNfreeSpec(const Pythia8::Info& info, int& nFreenProj, int& nFreepProj, int& nFreenTarg, int& nFreepTarg);
  /** @} */

  /** Pythia8 **/
  Pythia8::Pythia mPythia; //!

  /** @{
   * @name Configurations */
  /** configuration file to read **/
  std::string mConfig;
  /** ROOT script defining a Pythia8::UserHooks object */
  std::string mHooksFileName;
  /** Function in `mHooksFileName` to execute to return pointer to
   * Pythia8::UserHooks object */
  std::string mHooksFuncName;
  /** @} */

  UserFilterFcn mUserFilterFcn = [](Pythia8::Particle const&) -> bool { return true; };
  void initUserFilterCallback();

  bool mApplyPruning = false;

  ClassDefOverride(GeneratorPythia8, 1);

}; /** class GeneratorPythia8 **/

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATORPYTHIA8_H_ */
//
// EOF
//
