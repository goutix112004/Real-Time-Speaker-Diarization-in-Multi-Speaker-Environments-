import os
import sys
import argparse
import tempfile
import time
import random

# audio / recording
import sounddevice as sd
import soundfile as sf
import numpy as np
import torchaudio

# plotting & colors
import matplotlib.pyplot as plt
from colorama import init as colorama_init, Fore, Style

# diarization
try:
    from simple_diarizer.diarizer import Diarizer
except Exception as e:
    print("ERROR: missing 'simple-diarizer'. Install with: pip install simple-diarizer")
    raise

# transcription
import whisper

colorama_init(convert=True)

# Load Whisper model once
whisper_model = whisper.load_model("base")  # can change to "tiny", "small", "medium", "large"

# --------------------- helpers ---------------------
def rand_color_code():
    return random.choice([Fore.CYAN, Fore.MAGENTA, Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.WHITE])

def normalize_segments(segments):
    out = []
    for s in segments:
        if isinstance(s, dict):
            start = s.get("start", s.get("begin", s.get("stime", None)))
            end = s.get("end", s.get("stop", s.get("etime", None)))
            label = s.get("label", s.get("speaker", s.get("spk", None)))
            if start is None or end is None:
                vals = list(s.values())
                if len(vals) >= 3:
                    start, end, label = vals[0], vals[1], vals[2]
                else:
                    continue
        elif isinstance(s, (list, tuple)) and len(s) >= 3:
            start, end, label = s[0], s[1], s[2]
        else:
            continue
        try:
            out.append((float(start), float(end), str(label)))
        except:
            continue
    return out

def write_rttm(out_path, filename_id, segments):
    with open(out_path, "w", encoding="utf-8") as f:
        for start, end, label in segments:
            dur = max(0.0, end - start)
            spk = str(label).replace(" ", "_")
            line = f"SPEAKER {filename_id} 1 {start:.2f} {dur:.2f} <NA> <NA> {spk} <NA> <NA>\n"
            f.write(line)

def plot_timeline(segments, audio_path=None):
    if not segments:
        print("No segments to plot.")
        return
    labels = [s[2] for s in segments]
    unique = sorted(list(dict.fromkeys(labels)))
    spk2idx = {spk: i for i, spk in enumerate(unique)}
    colors = plt.cm.get_cmap("tab20", max(1, len(unique)))

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5*len(unique)))
    for (start, end, label) in segments:
        idx = spk2idx[label]
        ax.hlines(idx, start, end, linewidth=10, color=colors(idx))
        ax.text((start+end)/2, idx + 0.1, str(label), ha="center", va="bottom", fontsize=9)

    ax.set_yticks(list(spk2idx.values()))
    ax.set_yticklabels(list(spk2idx.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Diarization timeline{'' if audio_path is None else ': ' + os.path.basename(audio_path)}")
    plt.tight_layout()
    plt.show()

# --------------------- transcription helper ---------------------
def transcribe_segment(audio_path, start, end):
    waveform, sr = torchaudio.load(audio_path)
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    chunk = waveform[:, start_frame:end_frame]
    if chunk.shape[1] == 0:
        return ""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpname = tmp.name
    tmp.close()
    torchaudio.save(tmpname, chunk, sr)

    result = whisper_model.transcribe(tmpname, language="en")

    # Safe cleanup: ignore if file is still locked
    try:
        os.remove(tmpname)
    except PermissionError:
        pass

    return result["text"].strip()

# --------------------- main diarization functions ---------------------
def diarize_file(input_wav, num_speakers=None, embed_model="ecapa", cluster_method="sc"):
    print(f"\n🔊 Diarizing file: {input_wav}")
    diar = Diarizer(embed_model=embed_model, cluster_method=cluster_method)

    if num_speakers is not None:
        raw_segments = diar.diarize(input_wav, num_speakers=int(num_speakers))
    else:
        raw_segments = diar.diarize(input_wav)

    segments = normalize_segments(raw_segments)
    if not segments:
        print("⚠️  No speech segments found.")
        return []

    # Remap labels
    remap = {}
    segments_remapped = []
    for _, _, lab in segments:
        if lab not in remap:
            remap[lab] = f"Speaker_{len(remap)+1}"
    for s, e, lab in segments:
        segments_remapped.append((s, e, remap[lab]))

    # Print diarization + transcription
    print("\n--- Diarization + Transcription ---")
    for s, e, label in segments_remapped:
        text = transcribe_segment(input_wav, s, e)
        print(f"{label} ({s:.2f}s - {e:.2f}s): {text}")

    # Save RTTM
    base = os.path.basename(input_wav)
    file_id = os.path.splitext(base)[0]
    rttm_path = input_wav + ".rttm"
    write_rttm(rttm_path, file_id, segments_remapped)
    print(f"\n✅ RTTM saved to: {rttm_path}")
    print(f"Detected speakers: {sorted(list(set([lab for _,_,lab in segments_remapped])))}")

    try:
        plot_timeline(segments_remapped, audio_path=input_wav)
    except Exception as e:
        print("Plotting failed:", e)

    return segments_remapped

def diarize_mic(duration=10, samplerate=16000, num_speakers=None, **kwargs):
    print(f"\n🎤 Recording from microphone for {duration} seconds.")
    try:
        sd.default.samplerate = samplerate
        sd.default.channels = 1
        rec = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        for t in range(duration, 0, -1):
            sys.stdout.write(f"\r⏳ Recording... {t}s left ")
            sys.stdout.flush()
            time.sleep(1)
        sd.wait()
        print("\n✅ Recording finished.")
    except Exception as e:
        print("❌ Microphone recording failed:", e)
        return []

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpname = tmp.name
    tmp.close()
    sf.write(tmpname, rec, samplerate)
    print(f"Saved temporary file: {tmpname}")

    segments = diarize_file(tmpname, num_speakers=num_speakers, **kwargs)

    try:
        os.remove(tmpname)
    except:
        pass
    return segments

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="SpeechBrain-based diarization with transcription.")
    ap.add_argument("mode", nargs="+", help="'mic' to record, or path to audio file")
    ap.add_argument("--num_speakers", "-n", type=int, default=None)
    ap.add_argument("--embed_model", default="ecapa", choices=["ecapa","xvec"])
    ap.add_argument("--cluster", default="sc", choices=["sc","ahc"])
    args = ap.parse_args()

    if args.mode[0].lower() == "mic":
        duration = int(args.mode[1]) if len(args.mode) > 1 else 10
        diarize_mic(duration=duration, samplerate=16000, num_speakers=args.num_speakers,
                    embed_model=args.embed_model, cluster_method=args.cluster)
    else:
        input_file = args.mode[0]
        if not os.path.exists(input_file):
            print("ERROR: file not found:", input_file)
            sys.exit(1)
        diarize_file(input_file, num_speakers=args.num_speakers,
                     embed_model=args.embed_model, cluster_method=args.cluster)

if __name__ == "__main__":
    main()