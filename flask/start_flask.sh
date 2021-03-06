#!/bin/sh
# The above code is called a shebang, and it defines what program the sh script
# is interpreted with. The most common shebangs are #!/bin/sh and #!/bin/sh
# sh written in windows may not run in linux. To rectify this, start git bash
# and copy paste the contents of the windows file into a vim or nano edited
# file.

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
