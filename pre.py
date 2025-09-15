import soundfile as sf
import numpy as np
import librosa
import pyrnnoise
import ctypes

def main():
    in_path = "input.wav"
    out_path = "denoised.wav"

    # 🔥 Load audio as float32 in [-1, 1], resample to 16k mono
    audio, sr = librosa.load(in_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)

    rn = pyrnnoise.RNNoise(16000)
    rn.channels = 1  

    frame_size = 480
    denoised_audio = []

    # Access the underlying C lib + states
    lib = pyrnnoise.rnnoise.lib
    states = [pyrnnoise.rnnoise.create() for _ in range(rn.channels)]

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size].copy()  # float32 frame

        # Convert numpy -> ctypes pointer (float32 expected!)
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call C API directly
        lib.rnnoise_process_frame(states[0], ptr, ptr)

        # Append back
        denoised_audio.append(frame)

    if denoised_audio:
        denoised_audio = np.concatenate(denoised_audio).astype(np.float32)

        # Save as WAV (16-bit PCM)
        sf.write(out_path, denoised_audio, 16000, subtype="PCM_16")
        print(f"✅ Denoised file saved at: {out_path}")
    else:
        print("⚠️ No frames processed. Input may be too short.")

if __name__ == "__main__":
    main()
