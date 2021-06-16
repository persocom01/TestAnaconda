# FastAPI

A FastAPI testing playground.

## Installation

1. Create new python environment (optional)

You might want to install FastAPI and all the packages the app needs in a separate environment, which is what venv is for. To create a new environment,
open the project folder in cmd and type:

```
<!-- Non anaconda -->
python -m venv env_name
cd env_name/Scripts/activate

<!-- Ananconda -->
conda create -n env_name python==3.79
```

2. Install dependencies

With the new environment activated, install needed dependencies by entering the following:

```
<!-- Non anaconda -->
pip install fastapi
pip install python-multipart
pip install uvicorn[standard]

<!-- Ananconda -->
conda install -c conda-forge fastapi
conda install -c conda-forge python-multipart
conda install -c conda-forge uvicorn
```

`python-multipart` is needed by FastAPI to handle form data.
`uvicorn` is an ASGI server, which is needed to run a FastAPI app.

## Usage

To run a FastAPI app, in the FastAPI project folder enter:

```
uvicorn main:app --reload
```

`main` = is the name of the python file the FastAPI app resides in.
`app` = is the name of the FastAPI app.
`--reload` = reloads the server on code changes. It is recommended that this only be used during development.

## Writing the FastAPI file

A minimal FastAPI file looks like this:

```
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Hello World'}
```

The steps to write the file are as follows:

1. import FastAPI

```
from fastapi import FastAPI
```

2. Create FastAPI instance

```
app = FastAPI()
```

`app` here is the name of the app. It can be any valid name you desire and subsequently affects the command needed to run the file.

3. Create path operation

```
@app.get('/{var}')
```

`/{var}` = path the function applies to. Variables can be inserted into the path between `{}`. The path is attached to the url of the domain as follows:

```
https://example.com<path>
```

`.get` = is the operation, which can be one of the following HTTP methods:

* post - for creating data
* get - for reading data
* put - for updating data
* delete - for deleting data
* options
* head
* patch
* trace

4. Define the path operation function

The function can be `async` or not, depending on whether you will need `await` to use a library in the code.

###
