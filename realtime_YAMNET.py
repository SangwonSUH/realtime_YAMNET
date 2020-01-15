import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import keras

import yamnet.params as params
import yamnet.yamnet as yamnet_model
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet/yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet/yamnet_class_map.csv')

frame_len = int(params.SAMPLE_RATE * 1) # 1sec

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len)

cnt = 0
plt.ion()
while True:
    # data read
    data = stream.read(frame_len, exception_on_overflow=False)

    # byte --> float
    frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    # model prediction
    scores, melspec = yamnet.predict(np.reshape(frame_data, [1, -1]), steps=1)
    prediction = np.mean(scores, axis=0)

    # visualize input audio
    plt.imshow(melspec.T, cmap='jet', aspect='auto', origin='lower')
    plt.pause(0.001)
    plt.show()

    top5_i = np.argsort(prediction)[::-1][:5]

    # print result
    print('Current event:\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))

    # print idx
    print(cnt)
    cnt += 1

stream.stop_stream()
stream.close()
p.terminate()