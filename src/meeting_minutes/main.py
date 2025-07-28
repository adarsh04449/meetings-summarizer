#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from meeting_minutes.crews.meeting_minutes_crew.meeting_minutes_crew import MeetingMinutesCrew
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path


load_dotenv()
client = OpenAI()


class MeetingMinutesState(BaseModel):
    transcript: str = ""
    meeting_minutes: str = ""


class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()
    def transcribe_meeting(self):
        print("Generating Transcription")

        SCRIPT_DIR = Path(__file__).parent
        audio_path = str(SCRIPT_DIR / "EarningsCall.wav")

        audio = AudioSegment.from_file(audio_path, format="wav")

        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        full_transcription = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i+1}/{len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format = "wav")

            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", 
                    file=audio_file, 
                    response_format="text"
                )
                full_transcription += transcription.text + " "

        self.state.transcript = full_transcription

    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("generate meeting minutes")
        crew = MeetingMinutesCrew()

        inputs = {
            "transcript": self.state.transcript
        }
        meeting_minutes = crew.crew().kickoff(inputs)
        self.state.meeting_minutes = meeting_minutes

    @listen(generate_meeting_minutes)
    def create_draft_meeting_minutes(self):
        print("Creating meeting draft")

def kickoff():
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.kickoff()




if __name__ == "__main__":
    kickoff()
