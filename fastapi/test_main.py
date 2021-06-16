import pytest
from main import audiorecog
from fastapi import UploadFile

@pytest.mark.asyncio
def test_create_file():
    files = UploadFile(filename='file', file=open('./sent.wav', 'rb'))
    output = audiorecog(files)

    assert output == 'success'
