import wave
import math
import struct
import os

def generate_beep(filename, freq=440, duration=0.2, volume=0.5, num_beeps=1, gap=0.08):
    sample_rate = 44100
    output_path = os.path.join('static', 'audio', filename)
    with wave.open(output_path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        for _ in range(num_beeps):
            # Beep
            num_samples = int(duration * sample_rate)
            for i in range(num_samples):
                # Apply a slight fade in/out to avoid clicks
                fade_len = int(0.01 * sample_rate)
                fade = 1.0
                if i < fade_len:
                    fade = i / fade_len
                elif i > num_samples - fade_len:
                    fade = (num_samples - i) / fade_len
                
                value = int(fade * volume * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                f.writeframesraw(struct.pack('<h', value))
            
            # Gap
            num_gap_samples = int(gap * sample_rate)
            for _ in range(num_gap_samples):
                f.writeframesraw(struct.pack('<h', 0))

if __name__ == "__main__":
    os.makedirs(os.path.join('static', 'audio'), exist_ok=True)
    generate_beep('beep1.wav', freq=880, duration=0.08, num_beeps=3, gap=0.05)
    generate_beep('beep2.wav', freq=660, duration=0.12, num_beeps=2, gap=0.1)
    generate_beep('beep3.wav', freq=1200, duration=0.06, num_beeps=4, gap=0.04)
    print("Beep files generated in static/audio/")
