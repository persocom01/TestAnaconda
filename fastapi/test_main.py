import pytest
from fastapi import UploadFile
from app import create_file
import librosa

@pytest.mark.asyncio
async def test_create_file():
    files = UploadFile(filename='file', file=open('./test_wav.wav', 'rb'))
    output = await create_file(files)

    data, rate = librosa.load('audio.wav', sr=None)
    is_audio_correct = len(data) == 122880

    data, rate = librosa.load('noise_reduce.wav')
    is_noise_reduce_correct = len(data) == 56448

    is_audio_files_correct = all([is_audio_correct, is_noise_reduce_correct])

    assert output == {'message': 'transcribed done'} and is_audio_files_correct
