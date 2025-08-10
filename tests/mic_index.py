import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], 'IN', d['max_input_channels'])




