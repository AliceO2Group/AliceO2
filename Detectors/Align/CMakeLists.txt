# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(Align
               SOURCES  src/AliAlgAux.cxx
                        src/AliAlgConstraint.cxx
                        #src/AliAlgDet.cxx
                        #src/AliAlgDetHMPID.cxx
                        #src/AliAlgDetITS.cxx
                        #src/AliAlgDetTOF.cxx
                        #src/AliAlgDetTPC.cxx
                        #src/AliAlgDetTRD.cxx
                        src/AliAlgDOFStat.cxx
                        #src/AliAlgMPRecord.cxx
                        src/AliAlgPoint.cxx
                        #src/AliAlgRes.cxx
                        #src/AliAlgResFast.cxx
                        #src/AliAlgSens.cxx
                        #src/AliAlgSensHMPID.cxx
                        #src/AliAlgSensITS.cxx
                        #src/AliAlgSensTOF.cxx
                        #src/AliAlgSensTPC.cxx
                        #src/AliAlgSensTRD.cxx
                        #src/AliAlgSteer.cxx
                        #src/AliAlgTrack.cxx
                        src/AliAlgVol.cxx
                        #src/AliAlgVtx.cxx
                        #src/Mille.cxx
               PUBLIC_LINK_LIBRARIES O2::FrameworkLogger
               						 O2::ReconstructionDataFormats
                                     ROOT::Core
                                     ROOT::Geom
                                     ROOT::Gpad
                                     ROOT::Graf3d
                                     ROOT::MathCore
                                     ROOT::Matrix
                                     ROOT::Hist
                                     ROOT::RIO
                                     ROOT::Tree)

o2_target_root_dictionary(
  Align
  HEADERS #include/Align/AliAlgDet.h
          #include/Align/AliAlgDetITS.h
          #include/Align/AliAlgDetTOF.h
          #include/Align/AliAlgDetTPC.h
          #include/Align/AliAlgDetTRD.h
          #include/Align/AliAlgDetHMPID.h
          #include/Align/AliAlgMPRecord.h
          include/Align/AliAlgPoint.h
          #include/Align/AliAlgSens.h
          #include/Align/AliAlgSensITS.h
          #include/Align/AliAlgSensTOF.h
          #include/Align/AliAlgSensTPC.h
          #include/Align/AliAlgSensTRD.h
          #include/Align/AliAlgSensHMPID.h
          #include/Align/AliAlgSteer.h
          #include/Align/AliAlgTrack.h
          include/Align/AliAlgVol.h
          #include/Align/AliAlgVtx.h
          #include/Align/AliAlgRes.h
          #include/Align/AliAlgResFast.h
          include/Align/AliAlgConstraint.h
          include/Align/AliAlgDOFStat.h
          include/Align/AliAlgAux.h
          )
