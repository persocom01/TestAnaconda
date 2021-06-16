from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import noisereduce as nr
from scipy.io import wavfile
import librosa
import time
import uvicorn
import numpy as np

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/audiorecog')
def audiorecog(file: UploadFile = File(...)):
    start_time = time.time()
    audio_bytes = file.file.read()
    print('file: ', file)
    print('file: ', file.file)
    print('audio_bytes: ', len(audio_bytes))
    with open('./audio.wav', 'wb') as aud:
        aud.write(audio_bytes)

    data, rate = librosa.load('./audio.wav')
    noisy_part = data[0:10000]
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
    wavfile.write('./noise_reduce.wav', rate, np.asarray(reduced_noise, dtype=np.int16))

    print(f'It took {time.time() - start_time} seconds to convert blob to wav file and remove noise')

    return 'success'


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, log_level='debug')
