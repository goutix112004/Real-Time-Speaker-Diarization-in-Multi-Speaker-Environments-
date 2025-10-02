🎤 Microphone recording or process existing .wav files

🔊 Speaker diarization using simple-diarizer

🧠 Transcription with OpenAI’s Whisper ASR

🎨 Timeline visualization of speakers with Matplotlib

📝 Exports results in RTTM format (standard for diarization tasks)

⚡ Supports multiple embedding models (ecapa, xvec) and clustering methods (sc, ahc)
Requirements

Python 3.8+

simple-diarizer, whisper, torchaudio, matplotlib, sounddevice, soundfile
Output

Console log: Speaker-wise transcription with timestamps

RTTM file: Standard diarization output

Plot: Speaker timeline visualization
This project provides a Real-Time Speaker Diarization system that detects who spoke when in both audio files and live microphone recordings. It first performs diarization by segmenting the audio and grouping segments by speaker using embeddings (ECAPA or X-vector) and clustering methods (SC or AHC). Each speaker’s segments are then transcribed using Whisper ASR, producing speaker-wise text with timestamps. The output includes a console transcript, an RTTM file, and a speaker timeline plot, making it useful for meetings, interviews, and multi-speaker recordings.

To run the project, you need Python 3.8+, libraries such as simple-diarizer, openai-whisper, torchaudio, sounddevice, and FFmpeg installed for audio handling. Once dependencies are set up, you can analyze existing .wav files (e.g., python banana.py audio.wav --num_speakers 2) or record live from a microphone (e.g., python banana.py mic 10). The system then outputs clear, labeled transcripts and visualizations for multi-speaker environments.
