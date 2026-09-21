<!-- doxy
\page refrunSimExamplesMergeEventPools Example MergeEventPools
/doxy -->

This example demonstrates how to merge event pools using the dedicated `o2-generators-merge-evtpool`.
The pools to merge are listed in `pools.txt`. They can be local files or `alien://` paths, and a line can also point to
another local file list. AliEn pools are read directly from the storage elements, so a valid alien token is needed.
The merged pool is an ordinary evtpool.root file (by default), so it can be used with the `evtpool` generator.
