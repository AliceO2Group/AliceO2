#!/bin/sh

# one de per segmentation type
# remove the --sparse to get a very thorough testing
for de in 100 300 500 501 502 503 504 600 601 602 700 701 702 703 704 705 706 902 903 904 905; do
  o2-mchmapping-generate-pad-indices-impl4 --de ${de} --sparse >test_pad_indices_de${de}.json
done
