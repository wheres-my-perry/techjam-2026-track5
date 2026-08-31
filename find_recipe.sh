#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
C6="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv"
C7="$C6 data/manifests/canon_wukong.csv"
echo "TARGET canon6_train $(wc -l < data/manifests/canon6_train.csv)  val $(wc -l < data/manifests/canon6_val.csv)  test $(wc -l < data/manifests/canon6_test.csv)"
try () {  # $1 label, rest args
  L=$1; shift
  python -m scripts.build_canon6 --out-prefix data/manifests/zchk --exclude data/manifests/canon6_drop.txt "$@" > /tmp/zchk.log 2>&1
  T=$(wc -l < data/manifests/zchk_train.csv); V=$(wc -l < data/manifests/zchk_val.csv); E=$(wc -l < data/manifests/zchk_test.csv)
  M=NO; cmp -s data/manifests/zchk_train.csv data/manifests/canon6_train.csv && M=BYTE-IDENTICAL
  echo "  $L : train $T val $V test $E   $M"
  rm -f data/manifests/zchk_*.csv
}
try "6man cap45000        " --canon $C6 --cap-bucket 45000
try "7man cap45000        " --canon $C7 --cap-bucket 45000
try "6man cap0 equal-1    " --canon $C6 --cap-bucket 0 --equal-bucket -1
try "7man cap0 equal-1    " --canon $C7 --cap-bucket 0 --equal-bucket -1
try "6man cap45000 equal-1" --canon $C6 --cap-bucket 45000 --equal-bucket -1
echo RECIPE_DONE
