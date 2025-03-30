import speech_recognition as sr
from pathlib import Path
import os
from groq import Groq
import json
import librosa
import numpy as np
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class VoiceAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.client = Groq(api_key=os.getenv(api_key))

    def extract_audio_features(self, audio_file_path):
        """Extract audio features like pitch, volume, and speaking rate"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path)
            
            # Extract features
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/2])
            
            # Volume (RMS energy)
            volume = np.mean(librosa.feature.rms(y=y))
            
            # Speaking rate (zero crossing rate)
            speaking_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                "pitch": float(pitch_mean),
                "volume": float(volume),
                "speaking_rate": float(speaking_rate)
            }
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None

    def analyze_emotion(self, audio_file_path):
        """Analyze emotional factors from voice tonality and transcribed text"""
        # Get text transcription
        text = self.transcribe_audio(audio_file_path)
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio_file_path)
        
        if not audio_features:
            return None

        prompt = f"""
        Analyze the following voice characteristics and text for emotional indicators:
        
        Text transcription: {text}
        Voice features:
        - Pitch: {audio_features['pitch']} (higher values indicate higher pitch)
        - Volume: {audio_features['volume']} (higher values indicate louder speech)
        - Speaking rate: {audio_features['speaking_rate']} (higher values indicate faster speech)
        
        Please provide:
        1. Stress level (1-10)
        2. Confidence level (1-10)
        3. Anxiety level (1-10)
        4. Overall mood (positive/negative/neutral)
        5. Key emotional indicators based on both voice and text
        
        """

        response = self.client.chat.completions.create(
            model="Gemma2-9b-It",
            messages=[
                {"role": "system", "content": "You are an expert in voice emotion analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        # print(response)
        # print()
        # print(response.choices[0].message.content)

        # print(type(response.choices[0].message.content))

        try:
            emotion_data = response.choices[0].message.content
            print(type(emotion_data))
            # Include the transcription in the return value
            # emotion_data["transcription"] = text
            return emotion_data
        except Exception as e:
            print(e)
            return {
                "stress_level": 0,
                "confidence_level": 0,
                "anxiety_level": 0,
                "mood": "unknown",
                "emotional_indicators": [],
                "transcription": text
            }


    
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
    
    def process_audio(self, audio_path):
                """Process audio file and return comprehensive analysis"""
                if not os.path.exists(audio_path):
                    return {"error": "Audio file not found"}
                    
                try:
                    emotion_analysis = self.analyze_emotion(audio_path)
                    audio_features = self.extract_audio_features(audio_path)
                    transcription = self.transcribe_audio(audio_path)
                    
                    response = {
                        "emotion_analysis": emotion_analysis,
                        "audio_features": audio_features,
                        "transcription": transcription
                    }
                    print("Final Analysis:", json.dumps(response, indent=2))
                    return response
                except Exception as e:
                    return {"error": str(e)}
                    

if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    result = analyzer.process_audio("./harvard.wav")
    print(json.dumps(result, indent=2))