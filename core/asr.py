from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="medium", device="auto", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str):
        segments, info = self.model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True
        )
        text_parts, segs = [], []
        for s in segments:
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            text_parts.append(s.text.strip())
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": " ".join(text_parts).strip(),
            "segments": segs
        }
