# realtime_YAMNET
Simple real-time Sound Event Detector based on [YAMNET](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) and pyaudio.  
This is a toy project for SED, which you can analyze sound events with your own laptop mic in every second.  
The system shows the top five among the 521 target events.  

Thanks to the feature conversion functions implemented in tf_model itself, the code is extremely short.  
```
python realtime_YAMNET.py
```
And done.  
Enjoy.
