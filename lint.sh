#!/bin/bash

flake8 . --ignore=E501,W605,W504

code=$?

if [[ $code != 0 ]]; then
    echo
    echo "Please use 'yapf  --style pep8 --i --recursive .' to lint your code."
    exit 1
fi
