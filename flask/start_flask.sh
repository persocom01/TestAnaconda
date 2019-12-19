#!/bin/sh
export FLASK_APP=test_app
# development mode lets the server reload itself on code changes, and also sets
# FLASK_DEBUG=1
export FLASK_ENV=development
flask run

# For windows cmd?
# set FLASK_APP=test_app
# set FLASK_ENV=development
# flask run

# For windows powershell?
# $env:FLASK_APP=test_app
# $env:FLASK_ENV=development
# flask run
