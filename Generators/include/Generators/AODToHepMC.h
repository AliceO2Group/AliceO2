// Copyright 2023-2099 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Christian Holm Christensen <cholm@nbi.dk>

#ifndef ALICEO2_EVENTGEN_AODTOHEPMC_H_
#define ALICEO2_EVENTGEN_AODTOHEPMC_H_
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisManagers.h>
#include <Framework/Configurable.h>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/GenPdfInfo.h>
#include <HepMC3/GenHeavyIon.h>
#include <HepMC3/GenCrossSection.h>
#include <HepMC3/WriterAscii.h>
#include <fstream>

namespace o2
{
namespace eventgen
{
/**
 * Convert AOD tables of MC information into a HepMC event structure.
 *
 * The event structure is kept in memory.
 *
 * The conventions used here are
 *
 * - A @e track is a kinematics particle stored in the @c
 *   o2::aod::McParticles table (aliased as AODToHepMC::Tracks), and
 *   correspond to a track used during simulation transport.  These
 *   are of type @c o2::aod::McParticle aliased to AODToHepMC::Track.
 *
 *   - A @e generated track is a particle originally made by the event
 *     generator.  These can be recognised by @c
 *     Track::producedByGenerator()
 *
 * - A @e particle is a particle stored in the HepMC event structure
 *   of the class @c HepMC3::GenParticle.
 *
 * - A @e vertex is where a particle is either produced or disappears.
 *   A vertex with incoming particles will have a number of outgoing
 *   particles.  Thus a vertex is the @a production vertex for
 *   out-going particles of that vertex, and the @e end vertex of
 *   incoming particles of that vertex.  Vertexes are of the type @c
 *   HepMC3::GenVertex.
 *
 *   - The relationship between mother and daughter tracks in
 *     AODToHepMC::Tracks is encoded by indexes.
 *
 *     - At most two mothers can be stored per track.  If a track has
 *       no mothers, then both indexes are -1.  If a track has a
 *       single mother track, then both indexes are the same.
 *
 *     - A track can have any number of daughters.  The first daughter
 *       track is identified by the first index and the last daughter
 *       by the second stored index.  All tracks indexes with in the
 *       range from the first to the second index (both inclusive) are
 *       daughter tracks.  If a track has no daughters, then both
 *       indexes are -1.  If a track has one daughter, then both
 *       indexes are equal.  The number of daughters can be obtained
 *       by
 *
 *           last - first + 1
 *
 *       if both last and first are zero or greater.
 *
 * - An event header is the information stored in a row of @c
 *   o2::aod::McCollisions table (aliased to AODToHepMC::Headers).
 *   Each event has one such header (aliased as AODToHepMC::Header).
 *
 *   The header stores information on the event such as event number,
 *   weight, primary interaction point (IP), and so on.
 *
 *   In addition, auxiliary information (impact parameter,
 *   @f$N_{\mathrm{part}}@f$, and so on) may be stored separate tables.
 *
 *   - The table @c o2::aod::HepMCXSections (aliased to
 *     AODToHepMC::XSections) stores the event cross section and weight.
 *
 *   - The table @c o2::aod::HepMCPdfInfos (aliased to
 *     AODToHepMC::PdfInfos) stores the parton distribution function
 *     parameters used in the event.
 *
 *   - The table @c o2::aod::HepMCHeavyIons (aliased to
 *     AODToHepMC::PdfHeavyIons) stores the heavy-ion auxiliary
 *     information, such as the event impact parameter @f$b@f$,
 *     @f$N_{\mathrm{part}}@f$, @f$N_{\mathrm{coll}}@f$, and so on.
 *
 * - A HepMC event has a simple header which contains the event
 *   number, number of particles, and number of vertexes in the event.
 *
 *   Other event information is stored in specific structures.
 *
 *   - The event cross-section(s) and weight(s) is stored in a @c
 *     HepMC3::GenCrossSection (aliased to
 *     AODToHepMC::CrossSectionPtr) object.  The information to fill
 *     into that is read from AODToHepMC::XSections table
 *
 *   - The event parton distribution function parameters are stored in
 *     a @c HepMC3::GenPdfInfo (aliased to AODToHepMC::PdfInfoPtr)
 *     object.  The information to fill into that is read from
 *     AODToHepMC::PdfInfos table.
 *
 *   - The event heavy-ion parameters are stored in
 *     a @c HepMC3::GenHeavyIon (aliased to AODToHepMC::HeavyIonPtr)
 *     object.  The information to fill into that is read from
 *     AODToHepMC::HeavyIons table.
 *
 * The conversion is done as follows:
 *
 * - For all MC tracks, create the correspond particle
 *   (AODToHepMC::makeParticleRecursive)
 *
 *   - Check if we have already created the corresponding particle
 *     (AODToHepMC::getParticle). If so, then we go on to the next
 *     track.
 *
 *   - If we are asked to only keep generated tracks
 *     (AODToHepMC::mOnlyGen set by option @c --hepmc-only-generated),
 *     i.e., tracks that were posted by the event generator, and this
 *     track is not such a track (AODToHepMC::isIgnored), nor forced
 *     (argument @c force) then return nothing.  Note that mothers of
 *     particles that we keep are @e always made (@c force is set to
 *     true).
 *
 *   - If we do not have a particle yet for the track (look-up in
 *     AODToHepMC::mParticles) and it is not excluded, then create a
 *     particle that corresponds to the track, and store the mapping
 *     from track to particle (in AODToHepMC::mParticles).
 *
 *   - For all mother tracks of the current track, create the
 *     corresponding particle, following this algorithm (recursion,
 *     AODToHepMC::makeParticleRecursive).
 *
 *   - If a mother particle has an end vertex, set that vertex as the
 *     production vertex of the current particle.
 *
 *   - If no mother particle has an end vertex, and this particle is
 *     not a beam particle and it does have mothers, then create a
 *     vertex (AODToHepMC::makeVertex) corresponding to the track
 *     production position, and add this particle as an outgoing
 *     particle of that vertex.
 *
 *     - In this case, if some mother does not have an outgoing vertex,
 *       add that mother as an incoming particle to the created
 *       vertex.
 *
 *   - If the particle is a beam particle (status=4) then store this
 *     particle as a beam particle.
 *
 *   - If not a beam particle, and the particle has no mothers, mark
 *     this particle as an orphan.
 *
 * - Once we have made particle for all tracks, we flesh-out all
 *   particles. For all tracks (AODToHepMC::fleshOutParticle)
 *
 *   - If this track is ignored (AODToHepMC::isIgnored), or we have no
 *     particle (AODToHepMC::getParticle) corresponding to this track,
 *     go on to the next track.
 *
 *   - Get the end vertex of the particle.  If any. Set the candidate
 *     end vertex to this vertex, whether it exists or not.
 *
 *   - Then for each daughter track of the current track, check if it is
 *     consistent with the end vertex of the particle.
 *
 *     - Check if the dauther is ignored (AODToHepMC::isIgnored).  If so,
 *       move on to the next daughter track.
 *
 *     - Get the particle corresponding to the daughter track
 *       (AODToHepMC::getParticle.).  If no particle is found, move on
 *       to the next daughter track.
 *
 *     - Check that the daughter particle has an end vertex.  If it
 *       doesn't, mark it as a @a head-less particle.
 *
 *     - If the production vertex of the daughter doesn't match the end
 *       vertex of the current particle, or the current candidate end
 *       vertex, then issue a warning, and move on to the next daughter.
 *
 *     - Update the candidate end vertex to the daughter end vertex.
 *
 *   - After processing all daughter particles, and if we have no end
 *     vertex for the current particle, then
 *
 *     - if we found no candiate end vertex, and the particle is either a
 *       beam (status=4) or decayed (status=2) particle, issue a
 *       warning.
 *
 *      - if we do have a candidate end vertex, set that as the end vertex
 *        of the current particle.
 *
 *   - If, after this, the current particle does have an end vertex,
 *     loop over all daughters previsouly as head-less and set their
 *     production vertex to this end vertex.
 *
 * - At this point, we should have the particles in a proper HepMC
 *   event structure.
 *
 * - During simulation transport, the interaction point vertex (IP)
 *   may not be at (0,0,0,0).  Since some consumers of the the HepMC
 *   event structure may expect the IP to be at (0,0,0,0), we can
 *   recenter (AODMCToHepMC::recenter) all vertexes of the event.
 *   This is governed by the member AODMCToHepMC::mRecenter set by the
 *   option @c --hepmc-recenter
 *
 * - We can then fill in remaing information into the HepMC event header.
 *
 *   - The event number and weight is set from event header
 *     (AODToHepMC::makeHeader).
 *
 *   - The event cross section(s and weight(s) is set from
 *     AODToHepMC::CrossSections table.
 *     (AODToHepMC::makeXSection). If no AODToHepMC::CrossSections row
 *     is passed, then dummy values are filled in.
 *
 *   - The event parton distribution function parameters are set from
 *     AODToHepMC::PdfInfos table.  (AODToHepMC::makePdfInfo). If no
 *     AODToHepMC::PdfInfos row is passed, then dummy values are
 *     filled in.
 *
 *   - The event heavy-ion parameters are set from
 *     AODToHepMC::HeavyIons table.  (AODToHepMC::makeHeavyIon). If no
 *     AODToHepMC::HeavyIons row is passed, then dummy values are
 *     filled in.
 *
 * - Once all the above is done, we have a complete HepMC event
 *   (AODToHepMC::mEvent).  This event structure is kept in memory.
 *
 *   - Optionally (option @c --hepmc-dumb @e filename) we may write
 *     the events to a file (AODToHepMC::mOutput).  The event is still
 *     kept in memory.
 *
 *   - Clients of this class (e.g., o2::pwgmm::RivetWrapper) may access
 *     the event structure for further processing.
 *
 * The utility @c o2-aod-mc-to-hepmc will read in AODs and write out a
 * HepMC event file (plain ASCII).
 *
 */
struct AODToHepMC {
  /**
   * Group of configurables which will be added to an program that
   * uses this class.  Note that it is really the specialisation of
   * framework::OptionManager<AODToHepMC> that propagates the options
   * to the program.
   */
  struct : framework::ConfigurableGroup {
    /** Option for dumping HepMC event structures to disk.  Takes one
     * argument - the name of the file to write to. */
    framework::Configurable<std::string> dump{"hepmc-dump", "",
                                              "Dump HepMC event to output"};
    /** Option for only storing particles from the event generator.
     * Note, if a particle is stored down, then its mothers will also
     * be stored. */
    framework::Configurable<bool> onlyGen{"hepmc-only-generated", false,
                                          "Only export generated"};
    /** Use HepMC's tree parsing for building event structure */
    framework::Configurable<bool> useTree{"hepmc-use-tree", false,
                                          "Export as tree"};
    /** Floating point precision used when writing to disk */
    framework::Configurable<int> precision{"hepmc-precision", 8,
                                           "Export precision in dump"};
    /** Recenter event at IP=(0,0,0,0). */
    framework::Configurable<bool> recenter{"hepmc-recenter", false,
                                           "Recenter the events at (0,0,0,0)"};
  } configs;
  /**
   * @{
   * @name The containers we subscribe to
   */
  /** Alias of MC collisions table */
  using Headers = o2::aod::McCollisions;
  /** Alias of MC collisions table row */
  using Header = o2::aod::McCollision;
  /** Alias MC particles (tracks) table */
  using Tracks = o2::aod::StoredMcParticles_001;
  /** Alias MC particles (tracks) table row */
  using Track = typename Tracks::iterator;
  /** Alias auxiliary MC table of cross-sections */
  using XSections = o2::aod::HepMCXSections;
  /** Alias auxiliary MC table of parton distribution function
   * parameters */
  using PdfInfos = o2::aod::HepMCPdfInfos;
  /** Alias auxiliary MC table of heavy-ion parameters */
  using HeavyIons = o2::aod::HepMCHeavyIons;
  /** Alias row of auxiliary MC table of cross-sections */
  using XSection = typename XSections::iterator;
  /** Alias row of auxiliary MC table of parton distribution function
   * parameters */
  using PdfInfo = typename PdfInfos::iterator;
  /** Alias row of auxiliary MC table of heavy-ion parameters */
  using HeavyIon = typename HeavyIons::iterator;
  /** @} */
  /**
   * @{
   * @name Types from HepMC3
   */
  /** Alias HepMC four-vector */
  using FourVector = HepMC3::FourVector;
  /** Alias (smart-)pointer to HepMC particle */
  using ParticlePtr = HepMC3::GenParticlePtr;
  /** Alias (smart-)pointer to HepMC vertex */
  using VertexPtr = HepMC3::GenVertexPtr;
  /** Alias HepMC eventt structure */
  using Event = HepMC3::GenEvent;
  /** Alias (smart-)pointer to HepMC heavy-ion object */
  using HeavyIonPtr = HepMC3::GenHeavyIonPtr;
  /** Alias (smart-)pointer to HepMC cross section object */
  using CrossSectionPtr = HepMC3::GenCrossSectionPtr;
  /** Alias (smart-)pointer to HepMC parton distribution function
   * object */
  using PdfInfoPtr = HepMC3::GenPdfInfoPtr;
  /** Type used to map tracks to HepMC particles */
  using ParticleMap = std::map<long, ParticlePtr>;
  /** A container of pointers to particles */
  using ParticleVector = std::vector<ParticlePtr>;
  /** A container of pointers to vertexes */
  using VertexVector = std::vector<VertexPtr>;
  /** Alias of HepMC writer class */
  using WriterAscii = HepMC3::WriterAscii;
  /** The of pointer to HepMC writer class */
  using WriterAsciiPtr = std::shared_ptr<WriterAscii>;
  /** @} */
  /**
   * @{
   * @name HepMC3 objects
   */
  /** The result of processing */
  Event mEvent;
  /** Pointer to cross section-ion information */
  CrossSectionPtr mCrossSec = nullptr;
  /** Pointer to heavy-ion information */
  HeavyIonPtr mIon = nullptr;
  /** Pointer to parton distribution function information */
  PdfInfoPtr mPdf = nullptr;
  /** @} */
  /**
   * @{
   * @name Containers etc.
   */
  /** Maps tracks to particles */
  ParticleMap mParticles; //! Cache of particles
  /** List of vertexes made */
  VertexVector mVertices; //! Cache of vertices
  /** List of beam particles */
  ParticleVector mBeams; //! Cache of beam particles
  /** Particles without a mother */
  ParticleVector mOrphans; //! Cache of particles w/o mothers
  /** @} */
  /**
   * @{
   * @name Options and such
   */
  /** Output writer, if enabled */
  WriterAsciiPtr mWriter = nullptr;
  /** Current sequential event number */
  int mEventNo = 0;
  /** The last bunch crossing identifier */
  int mLastBC = -1;
  /** If true, only store particles from the generator */
  bool mOnlyGen = false;
  /** If true, use HepMC tree parser */
  bool mUseTree = true;
  /** Output stream if enabled */
  std::ofstream* mOutput = nullptr;
  /** Precision used on the output stream */
  int mPrecision = 16;
  /** If true, recenter IP to (0,0,0,0) */
  bool mRecenter = false;
  /** @} */

  /**
   * @{
   * @name Interface member functions
   */
  /**
   * Initialize the converter.  Sets internal parameters based on the
   * configurables.
   */
  virtual void init();
  /**
   * Process the collision header and tracks
   *
   * @param collision Header information
   * @param tracks    Particle tracks
   */
  virtual void process(Header const& collision, Tracks const& tracks);
  /**
   * Process collision header and HepMC auxiliary information
   *
   * @param collision Header information
   * @param xsections Cross-section table (possible no rows)
   * @param pdfs      Parton-distribution function table (possible no rows)
   * @param heavyions Heavy ion collision table (possible no rows)
   */
  virtual void process(Header const& collision,
                       XSections const& xsections,
                       PdfInfos const& pdfs,
                       HeavyIons const& heavyions);
  /**
   * End of run - closes output file if enabled.  This is called via
   * specialisation of o2::framework::OutputManager<AODToHepMC>.
   */
  virtual bool postRun()
  {
    enableDump("");
    return true;
  }
  /** @} */
 protected:
  /**
   * @{
   * @name Actual worker member functions
   */
  /**
   * Generate the final event, including fleshing out the vertexes,
   * and so on
   *
   * @param collision Header information
   * @param tracks Particle tracks
   */
  virtual void makeEvent(Header const& collision,
                         Tracks const& tracks);
  /**
   * Set the various fields in the header of the HepMC3::GenEvent
   * object
   *
   * @param header Header object
   */
  virtual void makeHeader(Header const& header);
  /**
   * Make cross-section information.  If no entry in the table,
   * then make dummy information
   */
  virtual void makeXSection(XSections const& xsection);
  /**
   * Make parton-distribition function information.  If no entry
   * in the table, then make dummy information
   */
  virtual void makePdfInfo(PdfInfos const& pdf);
  /**
   * Make heavy-ion collision information.  If no header given,
   * then fill in other reasonable values
   */
  virtual void makeHeavyIon(HeavyIons const& heavyion,
                            Header const& header);
  /**
   * This is supposed to make the beam particles from the information
   * available.  However, it seems like we really don't have enough
   * information, so for now we will do nothing here.  Perhaps the
   * user will be forced to provide that information - either via the
   * analyser configurables or somehow from somewhere else.
   */
  virtual void makeBeams(Header const& header, const VertexPtr ip) {}
  /**
   * Make all particles.  We loop through the MC particles, and for
   * each of them create that particle and any possible mother
   * particles (recursively).  This allows us to traverse the data
   * rather straight-forwardly.
   *
   * @param tracks The MC tracks
   */
  virtual void makeParticles(Tracks const& tracks);
  /**
   * Get particle corresponding to track @a no from particle cache
   *
   * @param  ref  Track reference
   * @return Pointer to HepMC3::GenParticle or null
   */
  virtual ParticlePtr getParticle(Track const& ref) const;
  /**
   * Check if we are ignoring this track
   *
   * @param track Track to check */
  virtual bool isIgnored(Track const& track) const;
  /**
   * Truely make a particle, and its mother particles if any.  We add
   * vertexes as needed here too.
   *
   * Note that this traverses the tree from the bottom up.  That is,
   * we check if a particle has any mothers, and if it does, we create
   * those.
   *
   * However, this can be a bit problematic, since the Kinematic tree
   * (and thus the McParticles table) only allows for at most 2
   * mothers.  In the case a vertex has 3 or more incoming particles,
   * then some of the intermediate particles will be lost.
   *
   * We remedy that problem by traversing the tree again, but this
   * time from the bottom up - that is, we look for daughters of all
   * particles, and if they are not registered with the out-going
   * vertex, then we reattch the parent to the incoming vertex of the
   * daughters.  For this to work, we need to map from track index to
   * HepMC3::GenParticle.
   *
   * @param track  Current track
   * @param tracks MC tracks
   * @param force  If true, do make the particle even if it would
   *               otherwise be skipped.
   *
   * @return Pointer to created particle.
   */
  virtual ParticlePtr makeParticleRecursive(Track const& track,
                                            Tracks const& tracks,
                                            bool force = false);
  /**
   * Generate a HepMC particle from a track.  Note that the job here
   * is simply to make the object.  The more complicated job of adding
   * the track to the tree is done above in makeParticleRecursive
   *
   * @param track MC track
   * @param mst   Mother status code - updated on return
   * @param force Force generation of particle even if onlyGen
   *
   * @returns Shared pointer to new HepMC particle
   */
  virtual ParticlePtr makeParticle(const Track& track,
                                   Int_t& motherStatus,
                                   bool force) const;
  /**
   * Generate vertex from production vertex of track
   *
   * @param track MC track
   *
   * @returns Shared pointer to new HepMC vertex
   */
  virtual VertexPtr makeVertex(const Track& track);
  /**
   * Visit all tracks, but this time look for daughters and
   * attach mothers as incoming particles if not already done.
   */
  virtual void fleshOut(Tracks const& tracks);
  /**
   * Flesh out a single particle
   *
   * @param track The track for which we should flesh out the
   *              corresponding particle.
   */
  virtual void fleshOutParticle(Track const& track, Tracks const& tracks);
  /**
   * Recenter event to (0,0,0,0).  This will use the vertex
   * information from the event header to translate all vertexes in
   * the event.
   *
   * @param header Event header
   */
  virtual void recenter(Header const& collision);
  /** @} */
  /**
   * Open dump output, or close if an empty string was given.
   *
   * @param dump
   */
  void enableDump(const std::string& dump);

}; /** class Generator **/

} // namespace eventgen

namespace framework
{
/**
 * This specialisation of o2::framework::OutputManager ensures that
 * we can call the post-processing routine of o2::eventgen::AODToHepMC
 * and thus ensure that the possible HepMC is written to disk.
 *
 * The O2 framework (via o2::framework::adoptAnalysisTask<T>) inspects
 * the members of the passed class (@c T) and creates
 * o2::framework::OutputManager callbacks for every member.  The
 * default template for this does nothing.
 *
 * Thus, to delegate a call to a member of the analysis task (of class
 * @c T), we can specialise the @c o2::framework::OutputManager
 * template on the @e member type.  We will then effectively have
 * call-backs for
 *
 * - @c appendOutput - when the task is constructed
 * - @c prepare - when a new set of data is recieved
 * - @c finalize - when a set of data has been processed
 * - @c postRun - when the run is over
 *
 * Concretely, we use the @c postRun to flush the HepMC data file
 * to disk.
 *
 * For this to work, the AODToHepMC object must be a member of the
 * "Task" class, e.g.,
 *
 * @code
 * struct Task {
 *   o2::eventgen::AODToHepMC mConverter;
 *   ...
 * };
 *
 * WorkflowSpec defineDataProcessing(ConfigContext const& cfg) {
 *   return WorkflowSpec{adaptAnalysisTask<Task>(cfg)};
 * }
 * @endcode
 */
template <>
struct OutputManager<eventgen::AODToHepMC> {
  /** Type of the target */
  using Target = eventgen::AODToHepMC;
  /** Called when task is constructed */
  static bool appendOutput(std::vector<OutputSpec>&, Target&, uint32_t) { return true; }
  /** Called when new data is received */
  static bool prepare(ProcessingContext&, Target&) { return true; }
  /** Called when all data has been received */
  static bool postRun(EndOfStreamContext&, Target& t) { return t.postRun(); }
  /** Called when the job finishes */
  static bool finalize(ProcessingContext&, Target& t) { return true; }
};

/**
 * Spacialisation to pull in configurables from the converter.
 *
 * Ideally, the converter should simply derive from ConfigurableGroup
 * and all should flow automatically, but that doesn't work for some
 * reason.
 *
 * For this to work, the AODToHepMC object must be a member of the
 * "Task" class, e.g.,
 *
 * @code
 * struct Task {
 *   o2::eventgen::AODToHepMC mConverter;
 *   ...
 * };
 *
 * WorkflowSpec defineDataProcessing(ConfigContext const& cfg) {
 *   return WorkflowSpec{adaptAnalysisTask<Task>(cfg)};
 * }
 * @endcode
 */
template <>
struct OptionManager<eventgen::AODToHepMC> {
  /** type of the target */
  using Target = eventgen::AODToHepMC;
  /** Called when the task is constructed */
  static bool
    appendOption(std::vector<o2::framework::ConfigParamSpec>& options,
                 Target& target)
  {
    OptionManager<ConfigurableGroup>::appendOption(options, target.configs);
    return true;
  }
  /** Called when options are processed */
  static bool
    prepare(o2::framework::InitContext& ic, Target& target)
  {
    OptionManager<ConfigurableGroup>::prepare(ic, target.configs);
    return true;
  }
};

} // namespace framework
} // namespace o2

#endif /* ALICEO2_EVENTGEN_GENERATOR_H_ */
// Local Variables:
//  mode: C++
// End:
