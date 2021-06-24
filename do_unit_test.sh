#!/bin/bash

src_dir=`pwd`
unit_test_dir=`pwd`/unit_test

mkdir tmp_build
cd tmp_build

nvcc ${src_dir}/gmm_matrix_support.cu -o libgmm_matrix_support.so -O3 -Xcompiler -fPIC -shared

for test_script in ${unit_test_dir}/*.py
do
    echo -n -e "${test_script##*/}:\n    "
    python ${test_script}
done

cd ${src_dir}
rm tmp_build -r
