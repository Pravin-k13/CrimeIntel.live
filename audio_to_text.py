from faster_whisper import WhisperModel
import os


def transcribe_audio(file_path, model_size="small"):
    """
    Transcribe English / Tamil / Mixed audio into English text
    with reduced hallucination and better real-world robustness.
    """

    # Load model (CPU optimized)
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )

    # Transcription settings (balanced for accuracy + stability)
    segments, info = model.transcribe(
        file_path,
        task="translate",                  # Translate Tamil -> English
        beam_size=5,                       # Better accuracy than 1
        temperature=0.2,                   # Slight randomness improves robustness
        condition_on_previous_text=False,  # Prevent hallucination chaining
        vad_filter=True,                   # Remove silence
        vad_parameters=dict(
            min_silence_duration_ms=300
        )
    )

    print(f"\nDetected Language: {info.language}")
    print(f"Audio Duration: {round(info.duration, 2)} seconds\n")

    full_text = ""

    for segment in segments:
        # Print debug info (optional – remove later if not needed)
        print(f"[{round(segment.start,2)}s - {round(segment.end,2)}s]")
        print("Text:", segment.text)
        print("Confidence (avg_logprob):", segment.avg_logprob)
        print("-" * 50)

        # Softer confidence filtering
        if segment.avg_logprob > -2.0:
            full_text += segment.text.strip() + " "

    return full_text.strip()


def audio_text_cvtr(audio_file):
    print(f'The audio file path is {audio_file}')
    if os.path.exists(audio_file):
        print("\nTranscribing...\n")
        text = transcribe_audio(audio_file, model_size="small")

        print("\nFinal Transcribed English Text:\n")
        print(text if text else "No valid speech detected.")
        return str(text)
    else:
        print("Audio file not found!")
        return "None"
