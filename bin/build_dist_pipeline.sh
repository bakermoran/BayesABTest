#!/bin/bash

echo 'MAKE SURE setup.py FILE IS UPDATED WITH NEWEST VERSION AND A RELEASE IS CREATED ON GITHUB'

sleep 5

python3 setup.py sdist bdist_wheel
twine upload dist/*
