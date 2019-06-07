o2_add_library(PHOSReconstruction SOURCES src/Cluster.cxx src/Clusterer.cxx
               src/ClustererTask.cxx
               PUBLIC_LINK_LIBRARIES O2::PHOSBase)

o2_target_root_dictionary(PHOSReconstruction HEADERS include/PHOSReconstruction/Cluster.h
               include/PHOSReconstruction/Clusterer.h
               include/PHOSReconstruction/ClustererTask.h)
