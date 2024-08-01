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

///
/// \file   VisualisationEventOpenGLSerializer.cxx
/// \brief  ROOT serialization
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEventOpenGLSerializer.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include <fairlogger/Logger.h>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace o2::event_visualisation
{

enum Header : uint8_t {
  version,
  runNumber,
  creationTimeUnused,
  firstTForbit,
  runType,
  trkMask,
  clMask,
  trackCount,
  clusterCount,
  phsCount,
  emcCount,
  primaryVertex,
  tfCounter,
  creationTimeLow,
  creationTimeHigh,
  last // number of fields
};

std::string detectors(const std::vector<std::string>& det, unsigned mask)
{
  std::string result;
  std::string delim;
  int bit = 1;
  for (const auto& i : det) {
    if (mask & bit) {
      result += delim + i;
      delim = ",";
    }
    bit = bit << 1;
  }
  return result;
}

const auto HEAD = "HEAD";
const auto TTYP = "TTYP";
const auto CELM = "CELM";
const auto TELM = "TELM";
const auto TIME = "TIME"; // track time
const auto SXYZ = "SXYZ"; // track start xyz
const auto CRGE = "CRGE"; // charge for track
const auto ATPE = "ATPE"; // angles: theta,phi,eta for track
const auto TGID = "TGID"; // track GID
const auto TPID = "TPID"; // track PID
const auto TXYZ = "TXYZ"; // track poinst x,y,z
const auto CXYZ = "CXYZ"; // track clusters x,y,z

const auto UXYZ = "UXYZ"; // global clusters x,y,z
const auto UGID = "UGID"; // global GID
const auto UTIM = "UTIM"; // global Time

const auto CALO = "CALO"; // calo phi,eta,enargy
const auto CALP = "CALP"; // calo PID
const auto CALG = "CALG"; // calo GID
const auto CALT = "CALT"; // calo PID

const auto FINE = "FINE"; //

void VisualisationEventOpenGLSerializer::toFile(const VisualisationEvent& event, std::string fileName)
{
  static const std::vector<std::string> det_coma = {
    "ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "ITS-TPC",
    "TPC-TOF", "TPC-TRD", "MFT-MCH", "ITS-TPC-TRD", "ITS-TPC-TOF", "TPC-TRD-TOF", "MFT-MCH-MID", "ITS-TPC-TRD-TOF", "ITS-AB", "CTP",
    "MCH-MID"};
  std::ostringstream buf;
  constexpr auto SIGSIZE = 512;
  unsigned char data[SIGSIZE];
  std::ofstream out(fileName, std::ios::out | std::ios::binary);
  // head --bytes 512 fileName.eve
  buf << "eve" << std::endl;
  buf << "version=1.00" << std::endl;
  buf << "run=" << event.getRunNumber() << std::endl;
  buf << "firstTForbit=" << event.getFirstTForbit() << std::endl;
  buf << "detectors=" << detectors(det_coma, event.getTrkMask()) << std::endl;
  buf << "preview='"
      << "dd if=thisFileName.eve of=thisFileName.png skip=20 bs=1 count=6000000" << std::endl;
  buf << std::string(SIGSIZE, ' ');
  memcpy((char*)&data[0], buf.str().c_str(), SIGSIZE);
  data[SIGSIZE - 2] = '\n';
  data[SIGSIZE - 1] = 0;
  out.write((char*)&data[0], SIGSIZE); // <----0 SIGN

  const auto trackNo = event.getTracksSpan().size();
  int phsCount = 0;
  int emcCount = 0;
  {
    for (const auto& calo : event.getCalorimetersSpan()) {
      if (calo.getSource() == o2::dataformats::GlobalTrackID::PHS) {
        phsCount++;
      }
      if (calo.getSource() == o2::dataformats::GlobalTrackID::EMC) {
        emcCount++;
      }
    }
  }

  // header
  {
    const auto chunkHEAD = createChunk(HEAD, Header::last * 4);
    const auto head = asUnsigned(chunkHEAD);
    head[Header::version] = event.mEveVersion;
    head[Header::runNumber] = event.getRunNumber();
    unsigned long creationTime = event.getCreationTime();
    head[Header::creationTimeLow] = creationTime;
    head[Header::creationTimeHigh] = creationTime / (1L << 32);
    ;
    head[Header::firstTForbit] = event.getFirstTForbit();
    head[Header::runType] = event.getRunType();
    head[Header::trkMask] = event.getTrkMask();
    head[Header::clMask] = event.getClMask();
    head[Header::trackCount] = event.getTrackCount();
    head[Header::clusterCount] = event.getClusterCount(); //  clusterno
    head[Header::phsCount] = phsCount;
    head[Header::emcCount] = emcCount;
    head[Header::primaryVertex] = event.getPrimaryVertex();
    head[Header::tfCounter] = event.getTfCounter();
    out.write(static_cast<char*>(chunkHEAD), chunkSize(chunkHEAD)); // <----1 HEAD
    free(chunkHEAD);
  }

  // information about number of track by type
  unsigned totalPoints = 0;
  unsigned totalClusters = 0;
  {
    const auto chunkTELM = createChunk(TELM, trackNo); // number of track points for each track
    const auto telm = asByte(chunkTELM);
    const auto chunkTGID = createChunk(TGID, 4 * trackNo); // number of track points for each track
    const auto tgid = asUnsigned(chunkTGID);
    const auto chunkTPID = createChunk(TPID, 4 * trackNo); // number of track points for each track
    const auto tpid = asUnsigned(chunkTPID);
    const auto chunkCELM = createChunk(CELM, trackNo); // number of track clusters for each track
    const auto celm = asByte(chunkCELM);

    const auto chunkTTYP = createChunk(TTYP, 4 * 27);
    const auto ttyp = asUnsigned(chunkTTYP);
    unsigned index = 0;
    for (const auto& track : event.getTracksSpan()) {
      tgid[index] = track.mBGID;
      tpid[index] = track.mPID;
      const auto ttypeidx = track.mBGID.getSource();
      ttyp[ttypeidx]++;
      totalPoints += track.getPointCount();     // here to pre-compute (performance)
      totalClusters += track.getClusterCount(); // here to pre-compute (performance)
      telm[index] = track.getPointCount();
      celm[index] = track.getClusterCount();
      index++;
    }
    out.write(static_cast<char*>(chunkTTYP), chunkSize(chunkTTYP)); // <----2 TTYP
    free(chunkTTYP);
    out.write(static_cast<char*>(chunkTELM), chunkSize(chunkTELM)); // <----3 TELM
    free(chunkTELM);
    out.write(static_cast<char*>(chunkCELM), chunkSize(chunkCELM)); // <----3 CELM
    free(chunkCELM);
    out.write(static_cast<char*>(chunkTGID), chunkSize(chunkTGID)); // <----3 GIND
    free(chunkTGID);
    out.write(static_cast<char*>(chunkTPID), chunkSize(chunkTPID)); // <----3 TPID (tracks pid)
    free(chunkTPID);
  }

  {
    const auto chunkTXYZ = createChunk(TXYZ, totalPoints * 4 * 3);
    const auto txyz = asFloat(chunkTXYZ);
    unsigned tidx = 0; // track elem (point coordinate)positions

    const auto chunkTIME = createChunk(TIME, trackNo * 4);
    const auto time = asFloat(chunkTIME);
    unsigned tno = 0; // track positions

    const auto chunkSXYZ = createChunk(SXYZ, trackNo * 4 * 3);
    const auto sxyz = asFloat(chunkSXYZ);
    unsigned sxyzidx = 0; // starting point track positions

    const auto chunkATPE = createChunk(ATPE, trackNo * 4 * 3);
    const auto atpe = asFloat(chunkATPE);
    unsigned atpeidx = 0; // starting point track positions

    const auto chunkCRGE = createChunk(CRGE, trackNo);
    const auto crge = asSignedByte(chunkCRGE);

    const auto chunkCXYZ = createChunk(CXYZ, totalClusters * 4 * 3);
    const auto cxyz = asFloat(chunkCXYZ);
    unsigned cidx = 0; // cluster positions

    for (const auto& track : event.getTracksSpan()) {
      time[tno] = track.getTime();
      crge[tno] = static_cast<signed char>(track.getCharge());
      tno++;
      sxyz[sxyzidx++] = track.getStartCoordinates()[0];
      sxyz[sxyzidx++] = track.getStartCoordinates()[1];
      sxyz[sxyzidx++] = track.getStartCoordinates()[2];

      atpe[atpeidx++] = track.getTheta();
      atpe[atpeidx++] = track.getPhi();
      atpe[atpeidx++] = track.mEta;

      for (unsigned i = 0; i < track.mPolyX.size(); i++) {
        txyz[tidx++] = track.mPolyX[i];
        txyz[tidx++] = track.mPolyY[i];
        txyz[tidx++] = track.mPolyZ[i];
      }
      for (unsigned i = 0; i < track.getClusterCount(); i++) {
        cxyz[cidx++] = track.getClustersSpan()[i].X();
        cxyz[cidx++] = track.getClustersSpan()[i].Y();
        cxyz[cidx++] = track.getClustersSpan()[i].Z();
      }
    }
    out.write(static_cast<char*>(chunkTXYZ), chunkSize(chunkTXYZ)); // <----4 TXYZ
    free(chunkTXYZ);
    out.write(static_cast<char*>(chunkCXYZ), chunkSize(chunkCXYZ)); // <----4 CXYZ
    free(chunkCXYZ);
    out.write(static_cast<char*>(chunkTIME), chunkSize(chunkTIME)); // <----4 TIME
    free(chunkTIME);
    out.write(static_cast<char*>(chunkSXYZ), chunkSize(chunkSXYZ)); // <----4 SXYZ
    free(chunkSXYZ);
    out.write(static_cast<char*>(chunkCRGE), chunkSize(chunkCRGE)); // <----4 CRGE
    free(chunkCRGE);
    out.write(static_cast<char*>(chunkATPE), chunkSize(chunkATPE)); // <----4 CRGE
    free(chunkATPE);
  }

  {
    const auto chunkUXYZ = createChunk(UXYZ, clusterCount * 4 * 3); // X,Y,Z
    const auto uxyz = asFloat(chunkUXYZ);
    const auto chunkUTIM = createChunk(UTIM, clusterCount * 4); // time
    const auto utim = asFloat(chunkUTIM);
    const auto chunkUGID = createChunk(UGID, clusterCount * 4); // time
    const auto ugid = asUnsigned(chunkUGID);
    unsigned idx = 0; // positions

    for (const auto& c : event.getClustersSpan()) {
      utim[idx / 3] = c.mTime;
      ugid[idx / 3] = serialize(c.mBGID);
      uxyz[idx++] = c.X();
      uxyz[idx++] = c.Y();
      uxyz[idx++] = c.Z();
    }
    out.write(static_cast<char*>(chunkUGID), chunkSize(chunkUGID)); //
    free(chunkUGID);
    out.write(static_cast<char*>(chunkUTIM), chunkSize(chunkUTIM)); //
    free(chunkUTIM);
    out.write(static_cast<char*>(chunkUXYZ), chunkSize(chunkUXYZ)); //
    free(chunkUXYZ);
  }

  {
    const auto chunkCALO = createChunk(CALO, (phsCount + emcCount) * 4 * 3); // phi, eta, energy
    const auto calo = asFloat(chunkCALO);
    const auto chunkCALP = createChunk(CALP, (phsCount + emcCount) * 4); // PID
    const auto calp = asUnsigned(chunkCALP);
    const auto chunkCALG = createChunk(CALG, (phsCount + emcCount) * 4); // PID
    const auto calg = asUnsigned(chunkCALG);
    const auto chunkCALT = createChunk(CALT, (phsCount + emcCount) * 4); // PID
    const auto calt = asFloat(chunkCALT);
    unsigned idx = 0; // positions

    for (const auto& c : event.getCalorimetersSpan()) {
      if (c.getSource() == o2::dataformats::GlobalTrackID::PHS) {
        calt[idx / 3] = c.getTime();
        calp[idx / 3] = serialize(c.getPID());
        calg[idx / 3] = serialize(c.getGID());
        calo[idx++] = c.getPhi();
        calo[idx++] = c.getEta();
        calo[idx++] = c.getEnergy();
      }
    }
    for (const auto& c : event.getCalorimetersSpan()) {
      if (c.getSource() == o2::dataformats::GlobalTrackID::EMC) {
        calt[idx / 3] = c.getTime();
        calp[idx / 3] = serialize(c.getPID());
        calg[idx / 3] = serialize(c.getGID());
        calo[idx++] = c.getPhi();
        calo[idx++] = c.getEta();
        calo[idx++] = c.getEnergy();
      }
    }

    out.write((char*)chunkCALO, chunkSize(chunkCALO)); //
    free(chunkCALO);
    out.write((char*)chunkCALP, chunkSize(chunkCALP)); //
    free(chunkCALP);
    out.write((char*)chunkCALG, chunkSize(chunkCALG)); //
    free(chunkCALG);
    out.write((char*)chunkCALT, chunkSize(chunkCALT)); //
    free(chunkCALT);
  }

  {
    const auto chunkFINE = createChunk(FINE, 0);
    out.write(static_cast<char*>(chunkFINE), chunkSize(chunkFINE)); // <----5 FINE
    free(chunkFINE);
  }
  out.close();
}

void* VisualisationEventOpenGLSerializer::createChunk(const char* lbl, unsigned size)
{
  const auto result = static_cast<unsigned char*>(calloc(4 * ((size + 3) / 4) + 8, 1));
  result[0] = lbl[0];
  result[1] = lbl[1];
  result[2] = lbl[2];
  result[3] = lbl[3];
  const auto uResult = (unsigned*)&result[4];
  *uResult = 4 * ((size + 3) / 4);
  return result;
}

unsigned VisualisationEventOpenGLSerializer::chunkSize(void* chunk)
{
  const auto uResult = (unsigned*)((char*)chunk + 4);
  return *uResult + 8;
}

long timestamp_from_filename(const std::string& s)
{
  const auto pos1 = s.find("tracks_");
  if (pos1 == std::string::npos) {
    return 0;
  }
  const auto pos2 = s.find('_', pos1 + 7);
  if (pos2 == std::string::npos) {
    return 0;
  }
  std::string::size_type sz; // alias of size_t
  const auto str_dec = s.substr(pos1 + 7, pos2 - pos1 - 7);
  const long li_dec = std::strtol(str_dec.c_str(), nullptr, 10);
  return li_dec;
}

bool VisualisationEventOpenGLSerializer::fromFile(VisualisationEvent& event, std::string fileName)
{
  std::filesystem::path inputFilePath{fileName};
  auto length = (long)std::filesystem::file_size(inputFilePath);
  if (length == 0) {
    return {}; // empty vector
  }
  std::vector<std::byte> buffer(length);
  std::ifstream inputFile(fileName, std::ios_base::binary);
  inputFile.read(reinterpret_cast<char*>(buffer.data()), length);
  inputFile.close();

  long position = 512;
  char type[5];
  type[0] = 0;
  type[4] = 0; // ending 0 for string

  auto trackCount = 0L;
  auto clusterCount = 0L;
  unsigned char* telm = nullptr;
  unsigned char* celm = nullptr;
  unsigned int* ttyp = nullptr;
  float* txyz = nullptr;
  float* cxyz = nullptr;

  float* time = nullptr;
  float* sxyz = nullptr;
  signed char* crge = nullptr;
  float* atpe = nullptr;
  unsigned* tgid = nullptr;
  unsigned* tpid = nullptr;

  float* calo = nullptr;
  unsigned* calp = nullptr;
  unsigned* calg = nullptr;
  float* calt = nullptr;

  unsigned* ugid = nullptr;
  float* utim = nullptr;
  float* uxyz = nullptr;

  unsigned phsCaloCount = 0;
  unsigned emcCaloCount = 0;

  while (true) {
    for (auto c = 0; c < 4; c++) {
      type[c] = (char)buffer.at(position + c);
    }
    auto* words = (unsigned*)(buffer.data() + position + 4);
    position = position + *words + 8;
    words++;
    if (std::string(type) == HEAD) {
      auto head = words;
      event.mEveVersion = head[Header::version];
      event.setRunNumber(head[Header::runNumber]);
      // event.setCreationTime(head[Header::creationTimeLow]+head[Header::creationTimeHigh]*(1L<<32));
      event.setCreationTime(timestamp_from_filename(fileName));
      event.setFirstTForbit(head[Header::firstTForbit]);
      event.setRunType((parameters::GRPECS::RunType)head[Header::runType]);
      event.setTrkMask((int)head[Header::trkMask]);
      phsCaloCount = head[Header::phsCount];
      emcCaloCount = head[Header::emcCount];
      event.setClMask((int)head[Header::clMask]);
      trackCount = head[Header::trackCount];
      clusterCount = head[Header::clusterCount];
      event.setPrimaryVertex(head[11]);
      event.setTfCounter(head[12]);
    } else if (std::string(type) == TELM) {
      telm = (unsigned char*)words;
    } else if (std::string(type) == CELM) {
      celm = (unsigned char*)words;
    } else if (std::string(type) == TTYP) {
      ttyp = (unsigned int*)words;
    } else if (std::string(type) == TIME) {
      time = (float*)words;
    } else if (std::string(type) == TXYZ) {
      txyz = (float*)words;
    } else if (std::string(type) == CXYZ) {
      cxyz = (float*)words;
    } else if (std::string(type) == SXYZ) {
      sxyz = (float*)words;
    } else if (std::string(type) == CRGE) {
      crge = (signed char*)words;
    } else if (std::string(type) == ATPE) {
      atpe = (float*)words;
    } else if (std::string(type) == TGID) {
      tgid = (unsigned*)words;
    } else if (std::string(type) == TPID) {
      tpid = (unsigned*)words;
    } else if (std::string(type) == UXYZ) {
      uxyz = (float*)words;
    } else if (std::string(type) == UGID) {
      ugid = (unsigned*)words;
    } else if (std::string(type) == UTIM) {
      utim = (float*)words;
    } else if (std::string(type) == CALO) {
      calo = (float*)words;
    } else if (std::string(type) == CALP) {
      calp = (unsigned*)words;
    } else if (std::string(type) == CALG) {
      calg = (unsigned*)words;
    } else if (std::string(type) == CALT) {
      calt = (float*)words;
    } else if (std::string(type) == FINE) {
      assert(telm != nullptr);
      assert(celm != nullptr);
      assert(ttyp != nullptr);
      assert(txyz != nullptr);
      assert(cxyz != nullptr);
      assert(time != nullptr);
      assert(sxyz != nullptr);
      assert(crge != nullptr);
      assert(atpe != nullptr);
      assert(tgid != nullptr);
      assert(tpid != nullptr);
      assert(uxyz != nullptr);
      assert(ugid != nullptr);
      assert(utim != nullptr);
      assert(calo != nullptr);
      assert(calp != nullptr);
      assert(calg != nullptr);
      assert(calt != nullptr);
      int ttypidx = 0; // tracks are stored in order ITS,TPC,... where ttyp cointains a number
      int txyzidx = 0; // coordinates
      int cxyzidx = 0; // coordinates
      // TRACKS
      for (auto t = 0; t < trackCount; t++) {
        while (ttyp[ttypidx] == 0) {
          ttypidx++;
        }
        ttyp[ttypidx]--;
        auto track = event.addTrack({.time = time[t],
                                     .charge = crge[t],
                                     .PID = (int)tpid[t],
                                     .startXYZ = {sxyz[3 * t + 0], sxyz[3 * t + 1], sxyz[3 * t + 2]},
                                     .phi = atpe[3 * t + 1],
                                     .theta = atpe[3 * t + 0],
                                     .eta = atpe[3 * t + 2],
                                     .gid = deserialize(tgid[t])});
        track->mPolyX.reserve(telm[t]);
        track->mPolyY.reserve(telm[t]);
        track->mPolyZ.reserve(telm[t]);
        while (telm[t]-- > 0) {
          track->mPolyX.push_back(txyz[txyzidx++]);
          track->mPolyY.push_back(txyz[txyzidx++]);
          track->mPolyZ.push_back(txyz[txyzidx++]);
        }
        track->mClusters.reserve(celm[t]);
        while (celm[t]-- > 0) {
          VisualisationCluster cluster(cxyz + cxyzidx, 0, 0);
          cxyzidx += 3;
          track->mClusters.push_back(cluster);
        }
      }

      // Clusters
      for (auto c = 0u; c < clusterCount; c++) {
        event.addGlobalCluster(uxyz + 3 * c, utim[c], deserialize(ugid[c]));
      }

      // Calos
      auto idx = 0;
      for (auto c = 0u; c < phsCaloCount + emcCaloCount; c++) {
        auto phi = calo[idx++];
        auto eta = calo[idx++];
        auto energy = calo[idx++];
        event.addCalo({.time = calt[c],
                       .energy = energy,
                       .phi = phi,
                       .eta = eta,
                       .PID = deserialize(calp[c]),
                       .gid = deserialize(calg[c])});
      }

      break;
    }
  }
  return false;
}

} // namespace o2::event_visualisation
