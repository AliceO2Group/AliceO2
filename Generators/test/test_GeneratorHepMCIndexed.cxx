// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// Tests for the indexed (seekable) access GeneratorHepMC uses to serve the events
/// of a HepMC3 ASCII file out of order.
///
/// The first test guards the assumption the whole scheme rests on: that a HepMC3
/// ASCII reader holds no state between events besides the stream, so that the stream
/// can be repositioned underneath it. That is an implementation detail of HepMC3, not
/// a documented contract, so it has to be re-checked against every version we build
/// against - if it ever stops holding, the generator would silently serve the wrong
/// events rather than fail.
/// @author M. Giacalone - September 2026
/// co-written with Claude Opus 5

#define BOOST_TEST_MODULE Test GeneratorHepMC indexed access
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <Generators/GeneratorService.h>
#include <CommonUtils/ConfigurableParam.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenRunInfo.h>
#include <HepMC3/ReaderAscii.h>
#include <HepMC3/ReaderAsciiHepMC2.h>
#include <HepMC3/WriterAsciiHepMC2.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace
{
/// the momentum of the outgoing particle of event i, distinct for every event so that
/// an event can be recognised from the particles it produced
double expectedPx(int i) { return 10. + i; }

/// writes a HepMC3 Asciiv3 file of nEvents events, each with two beam particles and a
/// pair of outgoing pions, and returns its name
std::string writeInput(const std::string& name, int nEvents)
{
  std::ofstream out(name);
  out << "HepMC::Version 3.02.05\n"
      << "HepMC::Asciiv3-START_EVENT_LISTING\n"
      << "W Default\n"
      << "T unit-test|indexed-access|\n";
  for (int i = 0; i < nEvents; ++i) {
    const double px = expectedPx(i);
    out << "E " << i + 1 << " 1 4\n"
        << "U GEV MM\n"
        << "W 1.0\n"
        << "P 1 0 2212 0.000000e+00 0.000000e+00 +6.500000e+03 6.500000e+03 9.383000e-01 4\n"
        << "P 2 0 2212 0.000000e+00 0.000000e+00 -6.500000e+03 6.500000e+03 9.383000e-01 4\n"
        << "V -1 0.000000e+00 0.000000e+00 0.000000e+00 [1,2]\n"
        << "P 3 -1 211 " << px << " 1.000000e+00 1.000000e+00 " << px + 5. << " 1.395700e-01 1\n"
        << "P 4 -1 -211 " << -px << " -1.000000e+00 -1.000000e+00 " << px + 5. << " 1.395700e-01 1\n";
  }
  out << "HepMC::Asciiv3-END_EVENT_LISTING\n";
  out.close();
  return name;
}

/// rewrites an Asciiv3 file in the HepMC2 IO_GenEvent format, and returns its name
std::string writeInputHepMC2(const std::string& asciiv3, const std::string& name)
{
  HepMC3::ReaderAscii in(asciiv3);
  HepMC3::WriterAsciiHepMC2 out(name);
  while (true) {
    HepMC3::GenEvent event;
    in.read_event(event);
    if (in.failed()) {
      break;
    }
    out.write_event(event);
  }
  in.close();
  out.close();
  return name;
}

/// everything of an event that has to survive being reached by a seek
std::string fingerprint(const HepMC3::GenEvent& event)
{
  std::string out = "n=" + std::to_string(event.event_number()) +
                    " np=" + std::to_string(event.particles().size()) +
                    " nv=" + std::to_string(event.vertices().size()) +
                    " nw=" + std::to_string(event.weights().size());
  char buf[128];
  for (const auto& p : event.particles()) {
    snprintf(buf, sizeof buf, " [%d,%d,%.9e,%.9e,%.9e,%.9e]", p->pid(), p->status(),
             p->momentum().x(), p->momentum().y(), p->momentum().z(), p->momentum().t());
    out += buf;
  }
  return out;
}
/// the entry the generator says the event was taken from
int announcedEntry(const o2::dataformats::MCEventHeader& header)
{
  const std::string key = "forwarding-generator_inputEventNumber";
  if (!header.hasInfo(key)) {
    return -1;
  }
  bool valid = false;
  auto entry = header.getInfo<int>(key, valid);
  return valid ? entry : -1;
}

/// the entry the served particles actually come from, read back from their momenta
int servedEntry(const std::vector<o2::MCTrack>& tracks)
{
  auto outgoing = std::find_if(tracks.begin(), tracks.end(),
                               [](const o2::MCTrack& t) { return t.GetPdgCode() == 211; });
  if (outgoing == tracks.end()) {
    return -1;
  }
  return (int)std::lround(outgoing->Px() - expectedPx(0));
}

/// points the HepMC generator at a file and configures how it serves its events
void configure(const std::string& file, int eventsToSkip = 0)
{
  o2::conf::ConfigurableParam::updateFromString(
    "GeneratorFileOrCmd.fileNames=" + file +
    ";HepMC.randomize=true;HepMC.roundRobin=true;HepMC.reshuffleOnRepeat=false"
    ";HepMC.rngseed=12345;HepMC.eventsToSkip=" +
    std::to_string(eventsToSkip));
}

} // namespace

namespace
{
/// reads every entry of a file by seeking to a recorded offset and requires the result to
/// be what reading the file from start to end gives
void checkSeekEquivalence(const std::string& file, bool hepmc2)
{
  auto makeReader = [hepmc2](std::shared_ptr<std::istream> stream) -> std::shared_ptr<HepMC3::Reader> {
    if (hepmc2) {
      return std::make_shared<HepMC3::ReaderAsciiHepMC2>(stream);
    }
    return std::make_shared<HepMC3::ReaderAscii>(stream);
  };

  // read the file from start to end, recording where every event begins
  std::vector<std::streamoff> offsets;
  std::vector<std::string> sequential;
  {
    auto stream = std::make_shared<std::ifstream>(file);
    BOOST_REQUIRE(stream->good());
    auto reader = makeReader(stream);
    while (true) {
      auto here = (std::streamoff)stream->tellg();
      HepMC3::GenEvent event;
      reader->read_event(event);
      if (reader->failed()) {
        break;
      }
      offsets.push_back(here);
      sequential.push_back(fingerprint(event));
    }
  }
  BOOST_REQUIRE_MESSAGE(!offsets.empty(), "no event indexed in " << file);

  // now read them by seeking, backwards, so that every jump goes against the stream
  auto stream = std::make_shared<std::ifstream>(file);
  BOOST_REQUIRE(stream->good());
  auto reader = makeReader(stream);
  // the run-level header sits ahead of the first event and has to be parsed once
  HepMC3::GenEvent header;
  reader->read_event(header);
  for (int entry = (int)offsets.size() - 1; entry >= 0; --entry) {
    stream->clear();
    stream->seekg(offsets[entry]);
    HepMC3::GenEvent event;
    reader->read_event(event);
    BOOST_REQUIRE_MESSAGE(!reader->failed(),
                          file << ": could not read entry " << entry << " by seeking");
    BOOST_CHECK_MESSAGE(fingerprint(event) == sequential[entry],
                        file << ": entry " << entry << " read by seeking differs from the "
                             << "sequential read; the HepMC3 reader can no longer be "
                             << "repositioned and GeneratorHepMC's indexed access is unsafe");
    BOOST_CHECK_MESSAGE(event.run_info() != nullptr,
                        file << ": entry " << entry << " lost its GenRunInfo");
  }
}
} // namespace

/// The ASCII readers must survive having their stream repositioned between events: reading
/// the entries by seeking to a recorded offset has to give exactly what reading the file
/// from start to end gives.  Checked for both formats the generator indexes.
BOOST_AUTO_TEST_CASE(hepmc3_reader_can_be_seeked)
{
  constexpr int nEvents = 20;
  auto asciiv3 = writeInput("test_GeneratorHepMCIndexed_seek.hepmc", nEvents);
  auto hepmc2 = writeInputHepMC2(asciiv3, "test_GeneratorHepMCIndexed_seek2.hepmc");

  checkSeekEquivalence(asciiv3, false);
  checkSeekEquivalence(hepmc2, true);

  std::remove(asciiv3.c_str());
  std::remove(hepmc2.c_str());
}

/// The generator must serve every event of the file exactly once per pass, start over in
/// roundRobin mode, and - the part that a wrong index would break silently - hand out the
/// event that actually sits at the entry it claims to be serving.  Driving this through
/// GeneratorService also exercises the configurable-parameter path the generator is
/// configured by in a real job.
BOOST_AUTO_TEST_CASE(generator_serves_a_permutation)
{
  constexpr int nEvents = 25;
  auto name = writeInput("test_GeneratorHepMCIndexed_gen.hepmc", nEvents);
  configure(name);

  o2::eventgen::GeneratorService service;
  service.initService("hepmc", "", o2::eventgen::NoVertexOption());

  // two full passes over the file, plus a bit
  std::vector<int> served;
  for (int i = 0; i < 2 * nEvents + 5; ++i) {
    auto event = service.generateEvent();
    auto entry = servedEntry(event.first);
    BOOST_REQUIRE_MESSAGE(entry >= 0 && entry < nEvents,
                          "event " << i << " belongs to no entry of the input");
    // the event handed out has to be the one the generator says it is serving; without
    // this the index could be wrong by any amount and every other check would still pass
    BOOST_CHECK_MESSAGE(announcedEntry(event.second) == entry,
                        "event " << i << ": the generator reports entry "
                                 << announcedEntry(event.second)
                                 << " but handed out the event stored at entry " << entry);
    served.push_back(entry);
  }

  // each pass uses every event of the file exactly once ...
  std::vector<int> all(nEvents);
  std::iota(all.begin(), all.end(), 0);
  std::vector<int> pass1(served.begin(), served.begin() + nEvents);
  std::vector<int> pass2(served.begin() + nEvents, served.begin() + 2 * nEvents);
  std::vector<int> sorted1 = pass1;
  std::vector<int> sorted2 = pass2;
  std::sort(sorted1.begin(), sorted1.end());
  std::sort(sorted2.begin(), sorted2.end());
  BOOST_CHECK(sorted1 == all);
  BOOST_CHECK(sorted2 == all);
  // ... the events are not simply served in file order ...
  BOOST_CHECK(pass1 != all);
  // ... with reshuffleOnRepeat off every pass repeats the first one ...
  BOOST_CHECK(pass2 == pass1);
  // ... and roundRobin keeps going past the end of the file
  BOOST_CHECK(std::equal(served.begin() + 2 * nEvents, served.end(), pass1.begin()));

  std::remove(name.c_str());
}

/// eventsToSkip has to leave the skipped entries out of the game entirely. This also pins
/// the index down to an absolute position in the file: an index off by any amount would
/// serve an entry from outside the requested range.
BOOST_AUTO_TEST_CASE(generator_honours_events_to_skip)
{
  constexpr int nEvents = 25;
  constexpr int toSkip = 18;
  auto name = writeInput("test_GeneratorHepMCIndexed_skip.hepmc", nEvents);
  configure(name, toSkip);

  o2::eventgen::GeneratorService service;
  service.initService("hepmc", "", o2::eventgen::NoVertexOption());

  std::vector<int> served;
  for (int i = 0; i < 2 * (nEvents - toSkip); ++i) {
    auto event = service.generateEvent();
    auto entry = servedEntry(event.first);
    BOOST_CHECK_MESSAGE(entry >= toSkip && entry < nEvents,
                        "event " << i << " came from entry " << entry
                                 << ", outside the requested range [" << toSkip << ", "
                                 << nEvents << ")");
    BOOST_CHECK_MESSAGE(announcedEntry(event.second) == entry,
                        "event " << i << ": the generator reports entry "
                                 << announcedEntry(event.second)
                                 << " but handed out the event stored at entry " << entry);
    served.push_back(entry);
  }

  std::vector<int> usable(nEvents - toSkip);
  std::iota(usable.begin(), usable.end(), toSkip);
  std::vector<int> pass1(served.begin(), served.begin() + (nEvents - toSkip));
  auto sorted = pass1;
  std::sort(sorted.begin(), sorted.end());
  BOOST_CHECK(sorted == usable);

  std::remove(name.c_str());
}

/// The same must hold when the input is in the HepMC2 IO_GenEvent format, which the
/// generator indexes with ReaderAsciiHepMC2 instead of ReaderAscii.
BOOST_AUTO_TEST_CASE(generator_reads_hepmc2)
{
  constexpr int nEvents = 25;
  auto asciiv3 = writeInput("test_GeneratorHepMCIndexed_h2src.hepmc", nEvents);
  auto name = writeInputHepMC2(asciiv3, "test_GeneratorHepMCIndexed_h2.hepmc");
  std::remove(asciiv3.c_str());
  configure(name);

  o2::eventgen::GeneratorService service;
  service.initService("hepmc", "", o2::eventgen::NoVertexOption());

  std::vector<int> served;
  for (int i = 0; i < nEvents; ++i) {
    auto event = service.generateEvent();
    auto entry = servedEntry(event.first);
    BOOST_REQUIRE_MESSAGE(entry >= 0 && entry < nEvents,
                          "event " << i << " belongs to no entry of the HepMC2 input");
    BOOST_CHECK_MESSAGE(announcedEntry(event.second) == entry,
                        "event " << i << ": the generator reports entry "
                                 << announcedEntry(event.second)
                                 << " but handed out the event stored at entry " << entry);
    served.push_back(entry);
  }
  std::vector<int> all(nEvents);
  std::iota(all.begin(), all.end(), 0);
  auto sorted = served;
  std::sort(sorted.begin(), sorted.end());
  BOOST_CHECK(sorted == all);
  BOOST_CHECK(served != all);

  std::remove(name.c_str());
}
