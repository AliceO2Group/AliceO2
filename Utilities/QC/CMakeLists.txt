add_subdirectory(QCCommon)

# We add it explicitely because it is a header only module.
include_directories(QCCommon/include)

add_subdirectory(QCProducer)
add_subdirectory(QCMerger)
add_subdirectory(QCViewer)
add_subdirectory(QCMetricsExtractor)
add_subdirectory(Workflow)
