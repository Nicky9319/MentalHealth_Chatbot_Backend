import os
import json
from groq import Groq

def transcribe_audio(self, audio_file_path):
    """
    Transcribes an audio file using the Groq client.

    Args:
        audio_file_path (str): Path to the audio file.
        model (str): Model to use for transcription. Default is "whisper-large-v3-turbo".
        prompt (str): Optional context or spelling prompt.
        response_format (str): Format of the response. Default is "verbose_json".
        timestamp_granularities (list): Optional list of timestamp granularities ("word", "segment", or both).
        temperature (float): Optional temperature setting for transcription. Default is 0.0.

    Returns:
        str: Transcription text.
    """
    # Initialize the Groq client

    model="whisper-large-v3-turbo"
    prompt=None 
    response_format="verbose_json" 
    timestamp_granularities=None 
    temperature=0.0


    client = Groq()

    # Open the audio file
    with open(audio_file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
            file=file,
            model=model,
            prompt=prompt,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            temperature=temperature
        )
        # Return the transcription text
        return transcription.text
