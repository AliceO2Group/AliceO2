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

/** @author Christian Holm Christensen <cholm@nbi.dk> */

#include <Framework/AnalysisHelpers.h>
#include <Framework/AnalysisTask.h>
#include <Generators/AODToHepMC.h>
//--------------------------------------------------------------------
/** Task to convert AOD MC tables into HepMC event structure
 *
 *  This assumes that the following tables are available on the input:
 *
 *  - @c o2::aod::McCollisions
 *  - @c o2::aod::McParticles
 *  - @c o2::aod::HepMCXSections
 *  - @c o2::aod::HepMCPdfInfos
 *  - @c o2::aod::HepMCHeavyIons
 *
 *  The application @c o2-sim-mcevent-to-aod publishes these tables.
 *
 *  Ideally, this application should work with the case where only
 *
 *  - @c o2::aod::McCollisions
 *  - @c o2::aod::McParticles
 *
 * This is selected by the option `--hepmc-no-aux`
 *
 * The thing to remember here, is that each task process is expected
 * to do a _complete_ job.  That is, a process _cannot_ assume that
 * another process has been called before-hand or will be called
 * later, for the same event in the same order.
 *
 * That is, each process will get _all_ events of a time-frame and
 * then the next process will get _all_ events of the time-frame.
 *
 * Processed do not process events piece-meal, but rather in whole.
 *
 */
struct AodToHepmc {
  /** Alias the converter type */
  using Converter = o2::eventgen::AODToHepMC;

  /** Our converter */
  Converter mConverter;

  /** @{
   * @name Container types */
  /** Alias converter header table type */
  using Headers = Converter::Headers;
  /** Alias converter header type */
  using Header = Converter::Header;
  /** Alias converter track table type */
  using Tracks = Converter::Tracks;
  /** Alias converter cross-section table type */
  using XSections = Converter::XSections;
  /** Alias converter cross-section type */
  using XSection = Converter::XSection;
  /** Alias converter parton distribution function table type */
  using PdfInfos = Converter::PdfInfos;
  /** Alias converter parton distribution function type */
  using PdfInfo = Converter::PdfInfo;
  /** Alias converter heavy-ions table type */
  using HeavyIons = Converter::HeavyIons;
  /** Alias converter heavy-ions type */
  using HeavyIon = Converter::HeavyIon;
  /** @} */

  /** Initialize the job */
  void init(o2::framework::InitContext& ic)
  {
    mConverter.init();
  }
  /** Processing of event to extract extra HepMC information
   *
   *  @param collision Event header
   *  @param tracks    Tracks of the event
   *  @param xsections Cross-section information
   *  @param pdf       Cross-section information
   *  @param heavyions Heavy ion (geometry) information
   */
  void process(Header const& collision,
               Tracks const& tracks,
               XSections const& xsections,
               PdfInfos const& pdfs,
               HeavyIons const& heavyions)
  {
    // Do not run this if --hepmc-no-aux was passed
    if (doPlain) {
      return;
    }
    LOG(debug) << "=== Processing everything ===";
    mConverter.startEvent();
    mConverter.process(collision,
                       xsections,
                       pdfs,
                       heavyions);
    mConverter.process(collision, tracks);
    mConverter.endEvent();
  }
  /** Processing of an event for particles only
   *
   *  @param collision  Event header
   *  @param tracks     Tracks of the event
   */
  void processPlain(Header const& collision,
                    Tracks const& tracks)
  {
    // Do not run this if --hepmc-no-aux was not passed
    if (not doPlain) {
      return;
    }

    LOG(debug) << "=== Processing only tracks ===";
    mConverter.startEvent();
    mConverter.process(collision, tracks);
    mConverter.endEvent();
  }
  /**
   * Make a process option.
   *
   * Instead of using the provided preprocessor macro, we instantise
   * the template directly here.  This is so that we can specify the
   * command line argument (@c --hepmc-no-aux) rather than to rely on an
   * auto-generated name (would be @c --processPlain).
   */
  decltype(o2::framework::ProcessConfigurable{&AodToHepmc::processPlain,
                                              "hepmc-no-aux", false,
                                              "Do not process auxiliary "
                                              "information"})
    doPlain = o2::framework::ProcessConfigurable{&AodToHepmc::processPlain,
                                                 "hepmc-no-aux", false,
                                                 "Do not process auxiliary "
                                                 "information"};
};

//--------------------------------------------------------------------
// This _must_ be included after our "customize" function above, or
// that function will not be taken into account.
#include <Framework/runDataProcessing.h>

//--------------------------------------------------------------------
using WorkflowSpec = o2::framework::WorkflowSpec;
using DataProcessorSpec = o2::framework::DataProcessorSpec;
using ConfigContext = o2::framework::ConfigContext;

/** Entry point of @a o2-sim-mcevent-to-hepmc */
WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  using o2::framework::adaptAnalysisTask;

  // Task: Two entry: header, tracks, and header, tracks, auxiliary
  return WorkflowSpec{adaptAnalysisTask<AodToHepmc>(cfg)};
}
//
// EOF
//
