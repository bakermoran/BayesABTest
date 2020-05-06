#!/bin/bash

# pycodestyle on BayesABTest/
for file in BayesABTest/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ pycodestyle $file"
  pycodestyle "$file"
done

# pydocstyle on BayesABTest/
for file in BayesABTest/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ pydocstyle $file"
  pydocstyle "$file"
done

# flake8 on BayesABTest/
for file in BayesABTest/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ flake8 $file"
  flake8 "$file"
done


# pycodestyle on tests/
for file in tests/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ pycodestyle $file"
  pycodestyle "$file"
done

# pydocstyle on tests/
for file in tests/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ pydocstyle $file"
  pydocstyle "$file"
done

# flake8 on tests/
for file in tests/*
do
  if [ ${file: -3} != ".py" ]
  then
    echo "skipping non .py file: $file"
    continue
  fi
  echo "+ flake8 $file"
  flake8 "$file"
done
